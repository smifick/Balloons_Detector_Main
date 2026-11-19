"""Start the local camera + SimpleBGC panel: python web_app.py."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--no-camera", action="store_true")
    args = parser.parse_args()
    import uvicorn
    from control_panel.server import create_app
    uvicorn.run(create_app(autostart_camera=not args.no_camera), host="127.0.0.1", port=args.port,
                access_log=False, log_level="info")


if __name__ == "__main__":
    main()
