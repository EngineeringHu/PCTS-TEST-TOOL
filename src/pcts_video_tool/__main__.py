import argparse
import json
import sys
from dataclasses import asdict

from . import __version__


def main():
    parser = argparse.ArgumentParser(description="PCTS 本地视频测量工具")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--inspect", metavar="VIDEO", help="检查视频帧数与时间戳，不启动界面")
    parser.add_argument("--self-test", metavar="JSON", help="运行安装包自检并写入JSON结果")
    args = parser.parse_args()
    if args.self_test:
        from .diagnostics import self_test

        sys.exit(self_test(args.self_test))
    if args.inspect:
        from .video import inspect_video, rate_summary

        info = inspect_video(args.inspect)
        data = asdict(info)
        data["frame_count"] = len(data.pop("times"))
        data["summary"] = rate_summary(info, 240)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return
    from PySide6.QtWidgets import QApplication

    from .run_logs import configure_logging
    from .ui import MainWindow

    configure_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("PCTS Video Tool")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
