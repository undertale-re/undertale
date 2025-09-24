import os

from bertviz import model_view
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin

from . import lib


class AttentionVisPlugin(base_plugin.TBPlugin):
    plugin_name = "visualize_attention_weights"

    def __init__(self, context):
        self.static_dir = os.path.join(os.path.dirname(__file__), "static")

    def is_active(self):
        return True

    def get_plugin_apps(self):
        return {
            "/index.js": self.serve_js,
            "/style.css": self.serve_css,
            "/attention.html": self.serve_attention,
        }

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(
            es_module_path="/index.js",
            tab_name="Visualize Attention",
        )

    @http_util.werkzeug.wrappers.Request.application
    def serve_js(self, request):
        with open(os.path.join(self.static_dir, "index.js")) as f:
            js = f.read()
        return http_util.werkzeug.Response(js, content_type="text/javascript")

    @http_util.werkzeug.wrappers.Request.application
    def serve_css(self, request):
        with open(os.path.join(self.static_dir, "style.css")) as f:
            css = f.read()
        return http_util.werkzeug.Response(css, content_type="text/css")

    @http_util.werkzeug.wrappers.Request.application
    def serve_attention(self, request):
        tokenizer_path = os.environ.get("UNDERTALE_TOKENIZER_PATH")
        checkpoint_path = os.environ.get("UNDERTALE_MASKEDLM_CHECKPOINT")

        tok = lib.load_tokenizer(tokenizer_path)
        model = lib.load_model(checkpoint_path)

        query = request.args.get("query", "").strip()

        attention, mask, predicted = lib.predict(
            query, tok, model, pretokenized=True, masked=True
        )
        attention = lib.remove_padded_tokens(mask, attention)
        html = model_view(attention, predicted, html_action="return")

        return http_util.werkzeug.Response(
            html.data,
            content_type="text/html; charset=utf-8",
            headers=[("X-Content-Type-Options", "nosniff")],
        )
