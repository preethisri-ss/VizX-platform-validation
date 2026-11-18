"""
VizX Phase 2 - Platform Review HTTP trigger (Azure Functions, Python v1 programming model)

POST body (JSON):
{
  "domain":          "Retail & E-commerce",            # optional, dropdown value
  "targetAudience":  "Consumers or buyers",            # optional, dropdown value
  "primaryGoal":     "Drive online sales of ...",      # optional, free-text prompt
  "geminiFileName":  "files/abc123"                    # str OR list of str (multiple screenshots)
                     or ["files/abc123", "files/def456"],
  "htmlContent":     "<html>...</html>"                # optional alternative to geminiFileName
}

Success (200):
  {"post_processed_response": {...}}                       # files supplied by the client
  {"post_processed_response": {...}, "gemini_file": {...}} # file created here from htmlContent

Errors: {"error": "<message>"} with 400 / 422 / 500 / 502 / 504.
"""

import json
import logging
import os
import tempfile

import azure.functions as func
from google import genai
from google.genai import errors as genai_errors

import shared_code.backend_logics_platform as backend_logics_platform

try:
    import pdfkit
except Exception:
    pdfkit = None

MAX_FILES_PER_REQUEST = 10
MAX_HTML_CHARS = 5_000_000


def _json_response(payload, status_code=200):
    return func.HttpResponse(json.dumps(payload), status_code=status_code,
                             mimetype="application/json")


def _error(message, status_code):
    return _json_response({"error": message}, status_code)


def _first(body, *keys):
    """Return the first non-empty value among the accepted key spellings."""
    for key in keys:
        value = body.get(key)
        if value not in (None, "", []):
            return value
    return None


def _parse_file_names(value):
    """Accept a single file name, a list of names, or a comma-separated string."""
    if value is None:
        return []
    if isinstance(value, str):
        names = value.split(",")
    elif isinstance(value, list):
        names = value
    else:
        raise ValueError("geminiFileName must be a string or a list of strings")
    cleaned = []
    for name in names:
        if not isinstance(name, str):
            raise ValueError("geminiFileName must be a string or a list of strings")
        name = name.strip()
        if name and name not in cleaned:
            cleaned.append(name)
    return cleaned


def _optional_text(value, field):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _html_to_gemini_file(html_content):
    """Render HTML to a temporary PDF and upload it to Gemini. Returns the uploaded file info."""
    if pdfkit is None:
        raise RuntimeError("pdfkit_unavailable")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpf:
            tmp_path = tmpf.name
        # Local file access is intentionally NOT enabled, so user-supplied HTML
        # cannot read files from the function host.
        pdfkit.from_string(html_content, tmp_path)

        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            raise EnvironmentError("GEMINI_API_KEY not set")
        client = genai.Client(api_key=gemini_api_key)
        uploaded_file = client.files.upload(file=tmp_path)
        return {"name": uploaded_file.name, "uri": getattr(uploaded_file, "uri", None)}
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("VizX platform review HTTP trigger processed a request.")

    # ---- Parse body -------------------------------------------------------
    try:
        req_body = req.get_json()
    except ValueError:
        return _error("Request body must be valid JSON", 400)
    if not isinstance(req_body, dict):
        return _error("Request body must be a JSON object", 400)

    try:
        domain = _optional_text(_first(req_body, "domain", "Domain"), "domain")
        target_audience = _optional_text(
            _first(req_body, "targetAudience", "target_audience", "audience"), "targetAudience")
        primary_goal = _optional_text(
            _first(req_body, "primaryGoal", "primary_goal", "prompt"), "primaryGoal")
        file_names = _parse_file_names(
            _first(req_body, "geminiFileName", "geminiFileNames", "geminifilename"))
        html_content = _optional_text(_first(req_body, "htmlContent"), "htmlContent")
    except ValueError as exc:
        return _error(str(exc), 400)

    logging.info("Domain is %s", domain)
    logging.info("Target audience is %s", target_audience)
    logging.info("Primary goal length is %s", len(primary_goal) if primary_goal else 0)
    logging.info("geminifilenames are %s", file_names)

    if len(file_names) > MAX_FILES_PER_REQUEST:
        return _error(f"A maximum of {MAX_FILES_PER_REQUEST} files can be reviewed per request", 400)

    # ---- Optional: build the Gemini file from raw HTML ---------------------
    uploaded_file_info = None
    if html_content and not file_names:
        if len(html_content) > MAX_HTML_CHARS:
            return _error("htmlContent is too large", 400)
        try:
            uploaded_file_info = _html_to_gemini_file(html_content)
            file_names = [uploaded_file_info["name"]]
        except RuntimeError as exc:
            if str(exc) == "pdfkit_unavailable":
                logging.error("pdfkit is not available. Install wkhtmltopdf and the pdfkit package.")
                return _error("Server cannot convert HTML to PDF: pdfkit or wkhtmltopdf not available", 500)
            logging.exception("HTML to PDF conversion failed")
            return _error("HTML to PDF conversion failed", 500)
        except Exception:
            logging.exception("Failed to convert or upload HTML content")
            return _error("Failed to prepare the HTML content for review", 500)

    if not file_names:
        return _error("Missing required parameter: geminiFileName or htmlContent", 400)

    # ---- Run the review; always clean up Gemini files ----------------------
    try:
        post_processed_response = backend_logics_platform.platform_gemini_access(
            domain, target_audience, primary_goal, file_names)
    except TimeoutError:
        logging.exception("Platform review timed out")
        return _error("The review took too long to complete. Please try again.", 504)
    except genai_errors.APIError as exc:
        logging.exception("Gemini API error")
        code = getattr(exc, "code", None)
        if code in (400, 403, 404):
            return _error("The uploaded file could not be found or accessed. Please upload it again.", 400)
        if code == 429:
            return _error("The review service is busy. Please try again shortly.", 503)
        return _error("The review service returned an error. Please try again.", 502)
    except ValueError:
        logging.exception("Platform review failed on file or model output")
        return _error("The uploaded file could not be processed. Please upload a valid PDF or image.", 422)
    except Exception:
        logging.exception("Unexpected error during platform review")
        return _error("Internal server error during platform review", 500)
    finally:
        # Files are deleted in every case so user screenshots never stay on Gemini
        try:
            backend_logics_platform.gemini_file_delete(file_names)
        except Exception:
            logging.exception("Failed to delete uploaded gemini files %s", file_names)

    response = {"post_processed_response": post_processed_response}
    if uploaded_file_info:
        response["gemini_file"] = uploaded_file_info
    return _json_response(response, 200)
