from shoplifting_system.app_factory import create_app

app = create_app()


if __name__ == '__main__':
    settings = app.config["runtime_settings"]
    app.run(host=settings.host, port=settings.port, debug=False)
