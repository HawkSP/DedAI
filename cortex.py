import websocket
import json
import ssl
import threading
import logging
import queue
import time
import traceback

class Cortex:
    def __init__(self, user_info, log_level=logging.INFO):
        """
        Constructor for the Cortex class.
        Establishes a connection with the Emotiv Cortex API.
        """
        # Store user info
        self.user_info = user_info

        # Initialize session status
        self.session_active = False

        # Create a queue for handling incoming data
        self.data_queue = queue.Queue()

        # Set up logging
        logging.basicConfig(level=log_level)

        # Create a connection to the Cortex API and authenticate
        self.reconnect()

        # Create a separate thread for processing data
        self.data_thread = threading.Thread(target=self.process_data, daemon=True)
        self.data_thread.start()

    def reconnect(self, backoff=1):
        """
        Reconnects and re-authenticates with the Cortex API.
        """
        try:
            self.ws = websocket.create_connection('wss://emotivcortex.com:54321',
                                                  sslopt={"cert_reqs": ssl.CERT_NONE})

            # Create a separate thread for receiving data
            self.thread = threading.Thread(target=self.receive_data, daemon=True)
            self.thread.start()

            self.authenticate()
        except Exception as e:
            logging.error(f"Error during reconnection: {e}")
            time.sleep(backoff)
            self.reconnect(backoff * 2)

    def receive_data(self):
        """
        Continuously receives data from the Cortex API.
        """
        while True:
            try:
                data = self.ws.recv()
                if data:
                    self.data_queue.put(data)
            except websocket.WebSocketConnectionClosedException:
                logging.info("Connection closed, attempting to reconnect...")
                self.reconnect()
                break

    def process_data(self):
        """
        Continuously processes data from the data queue.
        """
        while True:
            data = self.data_queue.get()
            if data:
                # Placeholder for data processing logic
                print(f"Data processed: {data}")

    def authenticate(self):
        """
        Authenticates with the Emotiv Cortex API.
        """
        try:
            if not self.ws.connected:
                raise Exception('WebSocket connection is not established.')

            auth = {
                "method": "authorize",
                "params": {
                    "clientId": self.user_info['clientId'],
                    "clientSecret": self.user_info['clientSecret']
                }
            }
            self.send_request(auth)
        except Exception as e:
            logging.error(f"Error during authentication: {e}")
            self.reconnect()

    def send_request(self, request):
        """
        Sends a JSON request to the Cortex API.
        """
        try:
            if not self.ws.connected:
                raise Exception('WebSocket connection is not established.')

            self.ws.send(json.dumps(request))
            result = json.loads(self.ws.recv())
            if 'error' in result and result['error'] is not None:
                logging.error('Error from Cortex: {}'.format(result['error']))
                raise Exception('Error from Cortex: {}'.format(result['error']))

            return result
        except Exception as e:
            logging.error(f"Error during request: {e}")
            self.reconnect()

    def create_session(self):
        """
        Creates a session for data collection.
        """
        try:
            if not self.ws.connected:
                logging.info("Connection not established, trying to reconnect...")
                self.reconnect()

            create = {
                "method": "createSession",
                "params": {
                    "cortexToken": self.user_info['cortexToken'],
                    "headset": self.user_info['headsetId'],
                    "status": "open"
                }
            }
            result = self.send_request(create)
            if 'result' in result and 'id' in result['result']:
                self.session_active = True

            return result
        except Exception as e:
            logging.error(f"Error during session creation: {e}")
            self.reconnect()

    def close_session(self):
        """
        Closes the currently active session.
        """
        if not self.ws.connected:
            logging.info("Connection not established, trying to reconnect...")
            self.reconnect()

        if not self.session_active:
            raise Exception('No active session to close.')

        close = {
            "method": "updateSession",
            "params": {
                "cortexToken": self.user_info['cortexToken'],
                "session": self.user_info['sessionId'],
                "status": "close"
            }
        }
        result = self.send_request(close)
        if 'result' in result and 'status' in result['result'] and result['result']['status'] == 'closed':
            self.session_active = False

        return result

    def subscribe_data_stream(self, streams):
        """
        Subscribes to the data stream from the headset.
        """
        subscribe = {
            "method": "subscribe",
            "params": {
                "cortexToken": self.user_info['cortexToken'],
                "session": self.user_info['sessionId'],
                "streams": streams
            }
        }
        return self.send_request(subscribe)

    def unsubscribe_data_stream(self, streams):
        """
        Unsubscribes from the data stream.
        """
        unsubscribe = {
            "method": "unsubscribe",
            "params": {
                "cortexToken": self.user_info['cortexToken'],
                "session": self.user_info['sessionId'],
                "streams": streams
            }
        }
        return self.send_request(unsubscribe)
