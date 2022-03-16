# encoding=utf-8
import uvicorn
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel
from main import NERapi

n = NERapi()
app = FastAPI()


class General(BaseModel):
    """传入文本数据"""

    data: str


@app.post("/")
async def index(request: General):
    """请求处理单元"""
    print("----------------------------")
    if len(request.data) == 0:
        return {}
    resp = ""
    resp = n.get_ner(request.data)
    return resp


if __name__ == '__main__':
    uvicorn.run("run:app", host="0.0.0.0", port=8899)



"""
nohup command >> ./test.log 2>&1 &
68908

查看运行的后台进程<只在当前终端生效，关闭终端后就无法看到了>
jobs -l

ps -aux | grep main.py
a: 显示所有程序
u: 以用户为主的格式来显示
x: 显示所有程序，不以终端机来区分
显示的最后一个是自己

ps -def | grep main.py | grep -v grep
去除自己

ps -aux | grep main.py | grep -v gerp | awk '{print $2}'
显示进程id

	
lsof -i:8090
查看使用某端口的进程

查看到进程id之后，使用netstat命令查看其占用的端口
netstat -ap|grep 8090
查看进程占用的端口

segmentation 分割
dimension 维度
optimizer 优化器
schema  架构
learning rate 学习率


python main.py \
    --emb_file=/home/wanglei/algorithm/ner/name_data/vec.txt \
    --train_file=/home/wanglei/algorithm/ner/name_data/example.train \
    --dev_file=/home/wanglei/algorithm/ner/name_data/example.dev \
    --test_file=/home/wanglei/algorithm/ner/name_data/example.test

flags.DEFINE_string("emb_file", os.path.join("data", "vec.txt"), "Path for pre_trained embedding")  # pre_training嵌入的路径
flags.DEFINE_string("train_file", os.path.join("data", "example.train"), "Path for train data")  # 数据集路径
flags.DEFINE_string("dev_file", os.path.join("data", "example.dev"), "Path for dev data")
flags.DEFINE_string("test_file", os.path.join("data", "example.test"), "Path for test data")
"""