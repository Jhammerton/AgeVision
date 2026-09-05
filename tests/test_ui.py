"""Browser-level checks for the upload and camera experience."""

import base64
import json
import re
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from playwright.sync_api import Playwright, expect, sync_playwright


class QuietStaticHandler(SimpleHTTPRequestHandler):
    def log_message(self, format_string, *args):
        pass


@pytest.fixture(scope="module")
def ui_server():
    static_dir = Path(__file__).parents[1] / "src" / "static"
    handler = partial(QuietStaticHandler, directory=static_dir)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    thread.join()


@pytest.fixture(scope="module")
def playwright_instance():
    with sync_playwright() as instance:
        yield instance


def launch_browser(playwright: Playwright):
    return playwright.chromium.launch(
        headless=True,
        args=[
            "--use-fake-device-for-media-stream",
            "--use-fake-ui-for-media-stream",
        ],
    )


def test_upload_preview_and_prediction(playwright_instance, ui_server) -> None:
    browser = launch_browser(playwright_instance)
    page = browser.new_page()
    page.route(
        "**/health",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "ok", "model_loaded": True}),
        ),
    )
    page.route(
        "**/api/v1/predict",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "predicted_age": 23.4,
                    "typical_error_years": 4.7,
                    "p90_error_years": 10.8,
                }
            ),
        ),
    )
    page.goto(ui_server)
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
        "AScY42YAAAAASUVORK5CYII="
    )
    page.locator("#file").set_input_files(
        {"name": "portrait.png", "mimeType": "image/png", "buffer": png}
    )

    expect(page.locator("#drop-zone")).to_have_class(re.compile(r"\bhas-preview\b"))
    expect(page.locator("#preview")).to_be_visible()
    page.locator("#submit").click()
    expect(page.locator("#result")).to_be_visible()
    expect(page.locator("#age-value")).to_have_text("23.4")
    browser.close()


def test_camera_panel_opens_when_camera_is_available(
    playwright_instance,
    ui_server,
) -> None:
    browser = launch_browser(playwright_instance)
    context = browser.new_context()
    context.grant_permissions(["camera"], origin=ui_server)
    page = context.new_page()
    page.route(
        "**/health",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"status": "ok", "model_loaded": True}),
        ),
    )
    page.goto(ui_server)
    page.locator("#camera-button").click()

    expect(page.locator("#camera-panel")).to_be_visible()
    page.locator("#cancel-camera").click()
    expect(page.locator("#camera-panel")).to_be_hidden()
    browser.close()
