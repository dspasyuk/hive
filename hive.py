import subprocess
import json
import os
import sys

class Hive:
    def __init__(self, node_path="node", script_path=None):
        """
        Initialize the Hive Python wrapper.
        
        Args:
            node_path (str): Path to the node executable.
            script_path (str): Path to the hive_cli.js script. Defaults to the same directory as this file.
        """
        if script_path is None:
            # Assume hive_cli.js is in the same directory
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hive_cli.js")
        
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Could not find hive_cli.js at {script_path}")

        try:
            self.process = subprocess.Popen(
                [node_path, script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr, # Pipe stderr to parent's stderr
                text=True,
                bufsize=1 # Line buffered
            )
        except FileNotFoundError:
             raise FileNotFoundError(f"Node.js executable not found at '{node_path}'. Please ensure Node.js is installed.")

    def _send_command(self, action, args=None):
        """
        Send a command to the Node.js subprocess and wait for response.
        """
        if self.process.poll() is not None:
             raise RuntimeError("Hive Node.js process has terminated unexpectedly.")

        command = json.dumps({"action": action, "args": args or {}})
        try:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
            
            response_line = self.process.stdout.readline()
            if not response_line:
                 raise RuntimeError("Hive Node.js process closed the connection (no response).")
            
            response = json.loads(response_line)
            
            if response.get("error"):
                raise RuntimeError(f"Hive Error: {response['error']}")
            
            return response.get("data")
        except BrokenPipeError:
            raise RuntimeError("Hive Node.js process pipe broken.")
        except json.JSONDecodeError:
             raise RuntimeError(f"Failed to decode response from Hive: {response_line}")

    def init(self, options=None):
        """
        Initialize the Hive database.
        
        Args:
            options (dict): Configuration options (dbName, storageDir, etc.)
        """
        return self._send_command("init", options)

    def add_file(self, file_path):
        """
        Add a file to the database.
        """
        return self._send_command("addFile", {"filePath": os.path.abspath(file_path)})

    def embed(self, input_data, type="text"):
        """
        Generate an embedding.
        
        Args:
            input_data (str): Text content or image path.
            type (str): "text" or "image".
        """
        return self._send_command("embed", {"input": input_data, "type": type})

    def find(self, query_vector, top_k=10):
        """
        Find similar documents.
        """
        return self._send_command("find", {"queryVector": query_vector, "topK": top_k})

    def insert_one(self, entry):
        """
        Insert a raw entry.
        """
        return self._send_command("insertOne", {"entry": entry})

    def delete_one(self, id):
        """
        Delete an item by ID.
        """
        return self._send_command("deleteOne", {"id": id})
    
    def update_one(self, query, entry):
        """
        Update an item.
        """
        return self._send_command("updateOne", {"query": query, "entry": entry})
    
    def remove_file(self, file_path):
        """
        Remove a file from the database.
        """
        return self._send_command("removeFile", {"filePath": os.path.abspath(file_path)})

    def close(self):
        """
        Terminate the Node.js process.
        """
        if self.process:
            self.process.stdin.close()
            self.process.terminate()
            self.process.wait()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
