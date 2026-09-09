"""Run: python -m src.visual.app --port 8000."""
import argparse
import json
import logging
import math
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException
from src.config import ROOT, DATA, LOGS, RESULTS, POSES, ensure_dirs
from src.visual.catalog import image_catalog
from src.visual.service import MattressService


def create_app(service=None):
    ensure_dirs()
    app=Flask(__name__,static_folder=str(ROOT/'src/visual/static'),static_url_path='/static')
    app.config['MAX_CONTENT_LENGTH']=8192
    service=service or MattressService()
    logging.basicConfig(level=logging.INFO,handlers=[logging.FileHandler(LOGS/'server.log',encoding='utf-8'),logging.StreamHandler()])

    @app.get('/')
    def index():
        return send_from_directory(app.static_folder,'index.html')

    @app.get('/api/catalog')
    def catalog():
        return jsonify({'people':sorted(set(service.raw.people.tolist())),'poses':POSES,
                        'images':image_catalog(),'sessions':service.sessions(),'shape':[44,24]})

    @app.get('/api/report')
    def report():
        return jsonify(service.report())

    @app.get('/api/projection')
    def projection():
        path=RESULTS/'identity_projection.json'
        return jsonify(json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'points':[]})

    @app.get('/api/frame')
    def frame():
        return jsonify(service.frame(request.args.get('person','SAI'),int(request.args.get('action',1)),int(request.args.get('frame',0))))

    @app.get('/api/replay')
    def replay():
        strength=float(request.args.get('strength',.5))
        if not math.isfinite(strength) or not 0<=strength<=1:
            raise ValueError('调节强度需在 0–1 之间')
        return jsonify(service.replay(int(request.args.get('session',0)),int(request.args.get('frame',0)),strength))

    @app.post('/api/enroll')
    def enroll():
        body=request.get_json()
        if not isinstance(body,dict) or not all(k in body for k in ('name','person','action')):
            raise ValueError('需提供 name、person、action')
        return jsonify(service.enroll(body['name'],body['person'],int(body['action'])))

    @app.get('/api/images/<image_id>')
    def images(image_id):
        item=next((i for i in image_catalog() if i['id']==image_id),None)
        if item is None:
            return jsonify({'error':'图像不存在'}),404
        return send_from_directory(DATA/'heapmap(Partial)',item['file'])

    @app.get('/api/export')
    def export():
        return send_from_directory(RESULTS,'metrics.json',as_attachment=True)

    @app.errorhandler(ValueError)
    @app.errorhandler(TypeError)
    def invalid(error):
        return jsonify({'error':str(error)}),400

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify({'error':error.description}),error.code

    return app


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8000)
    args=parser.parse_args()
    create_app().run(host=args.host,port=args.port,debug=False,threaded=True)
