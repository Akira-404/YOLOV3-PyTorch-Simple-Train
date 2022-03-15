import time

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/server1', methods=['POST'])
def server1():
    params = request.json if request.method == "POST" else request.args
    t1 = time.time()
    img = params['img']
    t2 = time.time()
    print(f'get the image time:{t2 - t1}s')

    result = {
        "code": 200,
        "message": 'success',
        "len": len(img)
    }

    return jsonify(result)


def run():
    app.config['JSON_AS_ASCII'] = False
    app.run(host='0.0.0.0', port=5000, use_reloader=False)


if __name__ == "__main__":
    run()
