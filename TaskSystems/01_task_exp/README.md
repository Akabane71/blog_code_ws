简单的任务队列模式

# 使用的组件
* Redis
* SQLLite

## 启动

```bash
docker run -d --name redis -p 6379:6379 redis:7
```

# 应用功能
1. 模拟gpt生成图片
> 这是一个十分耗时的任务，经过一个环节

2. 模拟多步骤耗时任务
> 多阶段的任务

# task system需要有的功能:
后端的基础路由 /api/v1

1. 创建任务
get 
/tasks


2. 查询任务状态
> 最新的状态进度
post
/tasks/{task_id}


3. 查询任务事件进度
> 会返回所以的步骤记录
get
/tasks/{task_id}/envents


4. 取消任务
/tasks/{task_id}/cancel


