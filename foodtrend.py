import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from app import app


if __name__ == '__main__':
    app.run(host='localhost', threaded=True)
