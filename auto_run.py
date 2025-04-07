import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import os

class PythonFileHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_modified = {}  # Keep track of last modified time for each file
        self.cooldown = 1  # seconds to wait between runs

    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            current_time = time.time()
            file_path = event.src_path

            # Check if enough time has passed since last run
            if file_path not in self.last_modified or current_time - self.last_modified[file_path] > self.cooldown:
                print(f"\n🔄 Changes detected in {os.path.basename(file_path)}")
                print("Running Python script...")
                try:
                    subprocess.run(['python', file_path], check=True)
                    print(f"✅ {os.path.basename(file_path)} completed successfully")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Error running {os.path.basename(file_path)}: {e}")
                self.last_modified[file_path] = current_time

def watch_directory():
    event_handler = PythonFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=False)
    observer.start()

    print("👀 Watching for changes in Python files")
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
