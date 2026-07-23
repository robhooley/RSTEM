_APP = None


def set_app(app):
    """
    Set the global ExpertPI application instance.

    This function stores the ExpertPI app reference for use by other RSTEM modules.
    Must be called during ExpertPI startup before any RSTEM functions are used.

    Parameters
    ----------
    app : object
        The ExpertPI application instance to be stored globally.

    Returns
    -------
    None
    """
    global _APP
    _APP = app


def get_app():
    """
    Get the global ExpertPI application instance.

    Retrieves the previously set ExpertPI app instance. Raises an error if
    the app has not been set, indicating that set_app() was not called during
    startup.

    Returns
    -------
    object
        The stored ExpertPI application instance.

    Raises
    ------
    RuntimeError
        If the ExpertPI app has not been set via set_app().
    """
    if _APP is None:
        raise RuntimeError(
            "ExpertPI app not set. Ensure ExpertPI calls RSTEM.app_context.set_app(app) during startup."
        )
    return _APP
