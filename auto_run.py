import os
import subprocess
import sys
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


def get_src_directory():
    """Get the source directory from command line arguments or use default."""
    return sys.argv[1] if len(sys.argv) > 1 else "./src"


class PythonFileHandler(FileSystemEventHandler):
    """Handler for Python file modifications."""

    def __init__(self):
        """Initialize the handler with a cooldown period."""
        self.last_modified = {}  # Keep track of last modified time for each file
        self.cooldown = 1  # seconds to wait between runs

    def on_modified(self, event):
        """Handle the event when a Python file is modified."""
        if event.src_path.endswith(".py"):
            current_time = time.time()
            file_path = event.src_path

            # Check if enough time has passed since last run
            if (
                file_path not in self.last_modified
                or current_time - self.last_modified[file_path] > self.cooldown
            ):
                # Clear console before running
                os.system("cls" if os.name == "nt" else "clear")

                print(f"\n🔄 Changes detected in {os.path.basename(file_path)}")
                print("Running Python script...")
                try:
                    subprocess.run(["python", file_path], check=True)
                    print(f"✅ {os.path.basename(file_path)} completed successfully")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Error running {os.path.basename(file_path)}: {e}")
                self.last_modified[file_path] = current_time


def watch_directory():
    """Watch the specified directory and its subdirectories for changes in Python files."""
    src_directory = get_src_directory()
    event_handler = PythonFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=src_directory, recursive=True)
    observer.start()

    print(
        f"👀 Watching for changes in Python files in {src_directory} and its subdirectories"
    )
    print("Press Ctrl+C to stop watching")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n👋 Stopped watching")

    observer.join()


if __name__ == "__main__":
    watch_directory()
