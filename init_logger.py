#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : __init__.py.py
# @Author: 王磊
# @Date  : 2021/08/25
# @Desc  : 
import sys
from loguru import logger
from pathlib import Path

LOG_FORMAT = '{level}: {time:YYYY-MM-DD HH:mm:ss,SSS}: [{file}:{line}] [{extra[logid]}] {message}'

def init_logger(logpath: Path):
    '''
    初始化 logger 系统
    '''
    config = {
        'handlers': [
            {
                'sink': sys.stdout,
                'format': LOG_FORMAT,
                'enqueue': True,
                'backtrace': True,
                'diagnose': False,
            },
            { # 服务日志，只记录 INFO 日志
                'sink': logpath.joinpath('server.log'),
                'format': LOG_FORMAT,
                'filter': lambda x: x['level'].name == 'INFO',
                'enqueue': True,
            },
            { # 详细日志，记录 DEBUG 及以上级别日志
                'sink': logpath.joinpath('grok.log'),
                'format': LOG_FORMAT,
                'level': 'DEBUG',
                'enqueue': True,
                'backtrace': True,
                'diagnose': False,
            },
        ],
        'extra': {
            'logid': '-',
        },
    }
    logger.configure(**config)
    logger.info('logger initialized')

    return logger
