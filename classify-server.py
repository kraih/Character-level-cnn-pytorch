# -*- coding: utf-8 -*-
"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import sys
import json
import glob
import argparse
import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import shutil
import http.server
import socketserver
import logging

from src.utils import *
from src.dataset import MyDataset
from src.character_level_cnn import CharacterLevelCNN


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Character-level convolutional networks for text classification""")
    parser.add_argument("-a", "--alphabet", type=str,
                        default="""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
    parser.add_argument("-m", "--max_length", type=int, default=1014)
    parser.add_argument("-f", "--feature", type=str, choices=["large", "small"], default="small",
                        help="small for 256 conv feature map, large for 1024 conv feature map")
    parser.add_argument("-i", "--input", type=str, default="input", help="path to input folder")
    args = parser.parse_args()
    return args

model = None

class ClassifyRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        test_set = MyDataset(None, data=post_data, max_length=1014)

        # it should be possible to fetch te_feature from test_set without going
        # through dataloader, but I was unable to find the magic call
        test_generator = DataLoader(test_set)
        te_feature, te_label = iter(test_generator).next()

        model.eval()
        if torch.cuda.is_available():
          te_feature = te_feature.cuda()
        with torch.no_grad():
          te_predictions = model(te_feature)
          out = F.softmax(te_predictions, 1)
          weight = torch.argmax(out[0])
          weighti = int(out[0][1].item() * 1000)
          print(True if weight == 1 else False, weighti)
          response = { 'license': (True if weight == 1 else False) }
          if weight == 1:
            response['confidence'] =  (weighti - 500) / 5
          else:
            response['confidence'] =  (500 - weighti) / 5
          self.wfile.write(json.dumps(response).encode('utf-8'))

  
def serve(opt):
    global model
    if torch.cuda.is_available():
      model = torch.load(opt.input)
    else:
      model = torch.load(opt.input, map_location='cpu')

    PORT = 5000

    Handler = ClassifyRequestHandler

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
       print("serving at port", PORT)
       try:
          httpd.serve_forever()
       except KeyboardInterrupt:
          pass
       httpd.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
       httpd.shutdown()
       httpd.server_close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    opt = get_args()
    serve(opt)
