"""Regression: empty SSE events must not count and buffering must not hide TTFT."""
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import threading
import time
import unittest
import benchmark

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):
        pass
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        self.send_response(200)
        self.send_header('Content-Type','text/event-stream')
        self.end_headers()
        def send(obj):
            self.wfile.write(('data: '+json.dumps(obj)+'\n\n').encode())
            self.wfile.flush()
        send({'choices':[{'delta':{'role':'assistant','content':''}}]})
        time.sleep(.03)
        send({'choices':[{'delta':{'content':'Hello'}}],'timings':{'prompt_n':12,'cache_n':0,'prompt_ms':20}})
        time.sleep(.04)
        send({'choices':[],'usage':{'prompt_tokens':12,'completion_tokens':1},'timings':{'prompt_n':12,'cache_n':0,'prompt_ms':20}})
        self.wfile.write(b'data: [DONE]\n\n')
        self.wfile.flush()

class Tests(unittest.TestCase):
    def test_stream_boundary_and_metadata(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        old=benchmark.URL
        benchmark.URL=f'http://127.0.0.1:{server.server_port}'
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            row=benchmark.stream_request({'messages':[]})
            self.assertIsNone(row['error'])
            self.assertEqual(row['text'],'Hello')
            self.assertGreaterEqual(row['ttft_ms'],25)
            self.assertGreater(row['total_ms']-row['ttft_ms'],25)
            self.assertLess(row['header_ms'],row['ttft_ms'])
            self.assertEqual(row['usage']['prompt_tokens'],12)
            self.assertEqual(row['timings']['cache_n'],0)
        finally:
            benchmark.URL=old
            server.shutdown()
            server.server_close()

if __name__=='__main__':
    unittest.main()
