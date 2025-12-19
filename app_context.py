_APP = None

def set_app(app):
    global _APP
    _APP = app

def get_app():
    if _APP is None:
        raise RuntimeError(
            "ExpertPI app not set. Ensure ExpertPI calls RSTEM.app_context.set_app(app) during startup."
        )
    return _APP