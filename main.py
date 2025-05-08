import asyncio
from fastapi import FastAPI, UploadFile, Form, File, Header, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import csv
import os
import json
from ultralytics import YOLO
import threading
# uvicorn main:app --reload

message_queue = asyncio.Queue()

main_event_loop = None

app = FastAPI()

# 添加 CORS 中间件，允许所有来源的请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有 HTTP 请求头
)

# 定义请求体结构
class DatasetQuery(BaseModel):
    page: int
    page_size: int
    task: Optional[str] = None  # 添加任务类型筛选参数

# 构造伪数据
def generate_fake_dataset(index: int) -> dict:
    # 根据索引确定任务类型
    task_types = ["detect", "classify", "segment", "pose", "obb"]
    task_type = task_types[index % len(task_types)]
    
    # 根据任务类型生成不同的数据集
    if task_type == "detect":
        return {
            "id": index,
            "creator_id": 1001,
            "name": f"SAR_Dataset_{index}",
            "task": "detect",
            "class_count": 5,
            "classes": [
                {"id": 0, "name": "ship"},
                {"id": 1, "name": "vehicle"},
                {"id": 2, "name": "building"},
                {"id": 3, "name": "airplane"},
                {"id": 4, "name": "bridge"},
            ],
            "description": "SAR目标检测数据集",
            "path": f"D:/datasets/sar_detect_{index}",
            "image_count": 1000,
            "train_count": 800,
            "val_count": 100,
            "test_count": 100,
            "size": 102400000,
            "upload_time": (datetime(2025, 3, 25, 12, 0, 0) + timedelta(days=index)).isoformat() + "Z",
        }
    elif task_type == "classify":
        return {
            "id": index,
            "creator_id": 1001,
            "name": f"SAR_Classify_Dataset_{index}",
            "task": "classify",
            "class_count": 10,
            "classes": [
                {"id": 0, "name": "urban"},
                {"id": 1, "name": "rural"},
                {"id": 2, "name": "forest"},
                {"id": 3, "name": "water"},
                {"id": 4, "name": "desert"},
                {"id": 5, "name": "mountain"},
                {"id": 6, "name": "coastline"},
                {"id": 7, "name": "agriculture"},
                {"id": 8, "name": "industrial"},
                {"id": 9, "name": "residential"},
            ],
            "description": "SAR图像分类数据集",
            "path": f"D:/datasets/sar_classify_{index}",
            "image_count": 2000,
            "train_count": 1600,
            "val_count": 200,
            "test_count": 200,
            "size": 204800000,
            "upload_time": (datetime(2025, 3, 25, 12, 0, 0) + timedelta(days=index)).isoformat() + "Z",
        }
    elif task_type == "segment":
        return {
            "id": index,
            "creator_id": 1001,
            "name": f"SAR_Segment_Dataset_{index}",
            "task": "segment",
            "class_count": 8,
            "classes": [
                {"id": 0, "name": "road"},
                {"id": 1, "name": "building"},
                {"id": 2, "name": "water"},
                {"id": 3, "name": "vegetation"},
                {"id": 4, "name": "bare_soil"},
                {"id": 5, "name": "vehicle"},
                {"id": 6, "name": "ship"},
                {"id": 7, "name": "airplane"},
            ],
            "description": "SAR图像分割数据集",
            "path": f"D:/datasets/sar_segment_{index}",
            "image_count": 1500,
            "train_count": 1200,
            "val_count": 150,
            "test_count": 150,
            "size": 153600000,
            "upload_time": (datetime(2025, 3, 25, 12, 0, 0) + timedelta(days=index)).isoformat() + "Z",
        }
    elif task_type == "pose":
        return {
            "id": index,
            "creator_id": 1001,
            "name": f"SAR_Pose_Dataset_{index}",
            "task": "pose",
            "class_count": 3,
            "classes": [
                {"id": 0, "name": "person"},
                {"id": 1, "name": "vehicle"},
                {"id": 2, "name": "aircraft"},
            ],
            "description": "SAR姿态估计数据集",
            "path": f"D:/datasets/sar_pose_{index}",
            "image_count": 800,
            "train_count": 640,
            "val_count": 80,
            "test_count": 80,
            "size": 81920000,
            "upload_time": (datetime(2025, 3, 25, 12, 0, 0) + timedelta(days=index)).isoformat() + "Z",
        }
    else:  # obb (Oriented Bounding Box)
        return {
            "id": index,
            "creator_id": 1001,
            "name": f"SAR_OBB_Dataset_{index}",
            "task": "obb",
            "class_count": 4,
            "classes": [
                {"id": 0, "name": "ship"},
                {"id": 1, "name": "airplane"},
                {"id": 2, "name": "bridge"},
                {"id": 3, "name": "vehicle"},
            ],
            "description": "SAR旋转框检测数据集",
            "path": f"D:/datasets/sar_obb_{index}",
            "image_count": 1200,
            "train_count": 960,
            "val_count": 120,
            "test_count": 120,
            "size": 122880000,
            "upload_time": (datetime(2025, 3, 25, 12, 0, 0) + timedelta(days=index)).isoformat() + "Z",
        }

fake_datasets = [generate_fake_dataset(i) for i in range(1, 51)]  # 50 条数据

@app.post("/v1/dataset/get/")
async def get_datasets(query: DatasetQuery, authorization: Optional[str] = Header(None)):
    print("📥 收到数据集列表请求:")
    print(f"- 页码: {query.page}, 每页数量: {query.page_size}")
    print(f"- 任务类型: {query.task}")
    print(f"- 授权头: {authorization}")

    # 根据任务类型筛选数据集
    filtered_datasets = fake_datasets
    if query.task:
        filtered_datasets = [dataset for dataset in fake_datasets if dataset["task"] == query.task]

    start = (query.page - 1) * query.page_size
    end = start + query.page_size
    paged_data = filtered_datasets[start:end]

    return JSONResponse(content={
        "code": 200,
        "message": "请求成功",
        "total": len(filtered_datasets),
        "page": query.page,
        "page_size": query.page_size,
        "datasets": paged_data
    })





@app.post("/v1/dataset/upload/")
async def upload_dataset(
    name: str = Form(...),
    task: str = Form(...),
    file: UploadFile = File(...),
    description: str = Form(''),
    authorization: str = Header(None)
):
    print("📝 接收到请求:")
    print(f"- 名称: {name}")
    print(f"- 任务类型: {task}")
    print(f"- 描述: {description}")
    print(f"- 文件名: {file.filename}")
    print(f"- 授权头: {authorization}")

    # 这里只是确认接收到了请求，不做实际保存处理
    return JSONResponse(content={
        "code": 200,
        "message": "数据集上传接口已接收请求",
        "filename": file.filename
    })

class ModelQuery(BaseModel):
    filter: Optional[dict] = {}
    page: int = 1
    page_size: int = 10

def generate_fake_model(index: int) -> dict:
    return {
        "id": 100 + index,
        "name": f"SAR_Detect_Model_{index}",
        "task": "detect",
        "status": "trained",
        "creator_id": 500,
        "create_time": (datetime(2025, 3, 24, 10, 0, 0) + timedelta(days=index)).isoformat() + "Z",
        "description": "SAR目标检测模型",
        "project_id": 1,
        "path": f"D:/models/sar_detect_v{index}.pth",
        "size": round(10.0 + index * 1000000, 2),  # 模拟不同大小
        "weight_type": "float32"
    }


fake_models = [generate_fake_model(i) for i in range(50)]

@app.post("/v1/model/get")
async def get_models(query: ModelQuery, authorization: Optional[str] = Header(None)):
    print("📥 收到模型列表请求:")
    print(f"- 筛选条件: {query.filter}")
    print(f"- 页码: {query.page}, 每页数量: {query.page_size}")
    print(f"- 授权头: {authorization}")

    filtered = [
        m for m in fake_models
        if all(m.get(k) == v for k, v in query.filter.items())
    ]

    start = (query.page - 1) * query.page_size
    end = start + query.page_size
    paged_models = filtered[start:end]

    return JSONResponse(content={
        "code": 200,
        "message": "请求成功",
        "total": len(filtered),
        "page": query.page,
        "page_size": query.page_size,
        "models": paged_models
    })

# 定义默认模型数据
default_models = [
    # YOLOv5系列
    {
        "id": 0,
        "name": "yolov5n",
        "task": "detect",
        "size": 3.9,
        "path": "D:/models/yolov5n.pt"
    },
    {
        "id": 1,
        "name": "yolov5s",
        "task": "detect",
        "size": 14.3,
        "path": "D:/models/yolov5s.pt"
    },
    {
        "id": 2,
        "name": "yolov5m",
        "task": "detect",
        "size": 40.8,
        "path": "D:/models/yolov5m.pt"
    },
    {
        "id": 3,
        "name": "yolov5l",
        "task": "detect",
        "size": 90.5,
        "path": "D:/models/yolov5l.pt"
    },
    {
        "id": 4,
        "name": "yolov5n_cls",
        "task": "classify",
        "size": 4.1,
        "path": "D:/models/yolov5n_classify.pt"
    },
    {
        "id": 5,
        "name": "yolov5s_cls",
        "task": "classify",
        "size": 15.2,
        "path": "D:/models/yolov5s_classify.pt"
    },
    {
        "id": 6,
        "name": "yolov5m_cls",
        "task": "classify",
        "size": 42.4,
        "path": "D:/models/yolov5m_classify.pt"
    },
    {
        "id": 7,
        "name": "yolov5n_seg",
        "task": "segment",
        "size": 4.5,
        "path": "D:/models/yolov5n_segment.pt"
    },
    {
        "id": 8,
        "name": "yolov5s_seg",
        "task": "segment",
        "size": 16.8,
        "path": "D:/models/yolov5s_segment.pt"
    },
    {
        "id": 9,
        "name": "yolov5m_seg",
        "task": "segment",
        "size": 43.7,
        "path": "D:/models/yolov5m_segment.pt"
    },
    
    # YOLOv8系列
    {
        "id": 10,
        "name": "yolov8n",
        "task": "detect",
        "size": 6.2,
        "path": "D:/models/yolov8n.pt"
    },
    {
        "id": 11,
        "name": "yolov8s",
        "task": "detect",
        "size": 15.6,
        "path": "D:/models/yolov8s.pt"
    },
    {
        "id": 12,
        "name": "yolov8m",
        "task": "detect",
        "size": 52.1,
        "path": "D:/models/yolov8m.pt"
    },
    {
        "id": 13,
        "name": "yolov8l",
        "task": "detect",
        "size": 87.4,
        "path": "D:/models/yolov8l.pt"
    },
    {
        "id": 14,
        "name": "yolov8n_cls",
        "task": "classify",
        "size": 5.5,
        "path": "D:/models/yolov8n_classify.pt"
    },
    {
        "id": 15,
        "name": "yolov8s_cls",
        "task": "classify",
        "size": 16.4,
        "path": "D:/models/yolov8s_classify.pt"
    },
    {
        "id": 16,
        "name": "yolov8m_cls",
        "task": "classify",
        "size": 50.7,
        "path": "D:/models/yolov8m_classify.pt"
    },
    {
        "id": 17,
        "name": "yolov8n_seg",
        "task": "segment",
        "size": 7.1,
        "path": "D:/models/yolov8n_segment.pt"
    },
    {
        "id": 18,
        "name": "yolov8s_seg",
        "task": "segment",
        "size": 18.2,
        "path": "D:/models/yolov8s_segment.pt"
    },
    {
        "id": 19,
        "name": "yolov8m_seg",
        "task": "segment",
        "size": 54.9,
        "path": "D:/models/yolov8m_segment.pt"
    },
    
    # YOLOv11系列
    {
        "id": 20,
        "name": "yolov11n",
        "task": "detect",
        "size": 7.3,
        "path": "D:/models/yolov11n.pt"
    },
    {
        "id": 21,
        "name": "yolov11s",
        "task": "detect",
        "size": 18.7,
        "path": "D:/models/yolov11s.pt"
    },
    {
        "id": 22,
        "name": "yolov11m",
        "task": "detect",
        "size": 55.8,
        "path": "D:/models/yolov11m.pt"
    },
    {
        "id": 23,
        "name": "yolov11l",
        "task": "detect",
        "size": 92.7,
        "path": "D:/models/yolov11l.pt"
    },
    {
        "id": 24,
        "name": "yolov11n_cls",
        "task": "classify",
        "size": 6.8,
        "path": "D:/models/yolov11n_classify.pt"
    },
    {
        "id": 25,
        "name": "yolov11s_cls",
        "task": "classify",
        "size": 19.5,
        "path": "D:/models/yolov11s_classify.pt"
    },
    {
        "id": 26,
        "name": "yolov11m_cls",
        "task": "classify",
        "size": 57.2,
        "path": "D:/models/yolov11m_classify.pt"
    },
    {
        "id": 27,
        "name": "yolov11n_seg",
        "task": "segment",
        "size": 8.2,
        "path": "D:/models/yolov11n_segment.pt"
    },
    {
        "id": 28,
        "name": "yolov11s_seg",
        "task": "segment",
        "size": 20.3,
        "path": "D:/models/yolov11s_segment.pt"
    },
    {
        "id": 29,
        "name": "yolov11m_seg",
        "task": "segment",
        "size": 58.6,
        "path": "D:/models/yolov11m_segment.pt"
    }
]

@app.post("/v1/model/getdefault")
async def get_default_models(query: ModelQuery, authorization: Optional[str] = Header(None)):
    print("📥 收到默认模型列表请求:")
    print(f"- 筛选条件: {query.filter}")
    print(f"- 页码: {query.page}, 每页数量: {query.page_size}")
    print(f"- 授权头: {authorization}")

    # 根据筛选条件过滤模型
    filtered = [
        m for m in default_models
        if all(m.get(k) == v for k, v in query.filter.items())
    ]

    # 分页处理
    start = (query.page - 1) * query.page_size
    end = start + query.page_size
    paged_models = filtered[start:end]

    return JSONResponse(content={
        "code": 200,
        "message": "请求成功",
        "total": len(filtered),
        "page": query.page,
        "page_size": query.page_size,
        "models": paged_models
    })

# 定义模型训练请求体结构
class ModelTrainRequest(BaseModel):
    model_id: int
    datasets_id: int
    train_details: Optional[dict] = {}

# 全局变量，用于跟踪训练状态
training_status = {}

@app.post("/v1/model/train")
async def model_train(request: ModelTrainRequest, authorization: Optional[str] = Header(None)):
    print("🚀 收到模型训练请求:")
    print(f"- 模型ID: {request.model_id}")
    print(f"- 数据集ID: {request.datasets_id}")
    print(f"- 训练细节: {request.train_details}")
    print(f"- 授权头: {authorization}")

    # 初始化训练状态
    training_status[request.model_id] = "training"
    
    # 这里只是确认接收到了请求，不做实际训练处理
    response_data = {
        "code": 200,
        "message": "模型训练请求已接收",
        "model_id": request.model_id,
        "datasets_id": request.datasets_id,
        "train_details": request.train_details
    }
    
    # 启动异步任务发送训练结果
    # 可以选择使用真实训练函数或示例函数
    use_real_training = True
    
    if use_real_training:
        # 使用真实训练函数
        # asyncio.create_task(train_yolo_model(request.model_id, request.datasets_id, request.train_details))
        asyncio.create_task(train_example(request.model_id, request.train_details))
    else:
        # 使用示例训练结果
        asyncio.create_task(send_example_training_results(request.model_id, request.train_details))
    
    return JSONResponse(content=response_data)

# # 暂停训练接口
# @app.post("/v1/model/train/suspend")
# async def suspend_training(model_id: int, authorization: Optional[str] = Header(None)):
#     print(f"⏸️ 收到暂停训练请求: 模型ID {model_id}")
#     print(f"- 授权头: {authorization}")
    
#     if model_id in training_status:
#         training_status[model_id] = "suspend"
#         return JSONResponse(content={
#             "code": 200,
#             "message": "训练已暂停",
#             "model_id": model_id
#         })
#     else:
#         return JSONResponse(content={
#             "code": 404,
#             "message": "未找到指定的训练任务",
#             "model_id": model_id
#         })

# # 恢复训练接口
# @app.post("/v1/model/train/resume")
# async def resume_training(model_id: int, authorization: Optional[str] = Header(None)):
#     print(f"▶️ 收到恢复训练请求: 模型ID {model_id}")
#     print(f"- 授权头: {authorization}")
    
#     if model_id in training_status and training_status[model_id] == "suspend":
#         training_status[model_id] = "training"
#         return JSONResponse(content={
#             "code": 200,
#             "message": "训练已恢复",
#             "model_id": model_id
#         })
#     elif model_id not in training_status:
#         return JSONResponse(content={
#             "code": 404,
#             "message": "未找到指定的训练任务",
#             "model_id": model_id
#         })
#     else:
#         return JSONResponse(content={
#             "code": 400,
#             "message": "训练任务未处于暂停状态",
#             "model_id": model_id
#         })

# # 取消训练接口
# @app.post("/v1/model/train/cancel")
# async def cancel_training(model_id: int, authorization: Optional[str] = Header(None)):
#     print(f"❌ 收到取消训练请求: 模型ID {model_id}")
#     print(f"- 授权头: {authorization}")
    
#     if model_id in training_status:
#         # 从训练状态中移除，这会导致训练任务退出
#         del training_status[model_id]
#         return JSONResponse(content={
#             "code": 200,
#             "message": "训练已取消",
#             "model_id": model_id
#         })
#     else:
#         return JSONResponse(content={
#             "code": 404,
#             "message": "未找到指定的训练任务",
#             "model_id": model_id
#         })



# WebSocket连接管理器
class ConnectionManager:
    def __init__(self):
        # 使用字典存储连接，键为模型ID，值为WebSocket连接列表
        self.active_connections: dict = {}

    async def connect(self, websocket: WebSocket, model_id: int):
        await websocket.accept()
        if model_id not in self.active_connections:
            self.active_connections[model_id] = []
        self.active_connections[model_id].append(websocket)

    def disconnect(self, websocket: WebSocket, model_id: int):
        if model_id in self.active_connections:
            self.active_connections[model_id].remove(websocket)
            # 如果该模型ID没有连接了，则删除该键
            if not self.active_connections[model_id]:
                del self.active_connections[model_id]

    async def broadcast_to_model(self, message: str):
        message_data = json.loads(message)
        model_id = message_data.get("model_id")
        if model_id in self.active_connections:
            print(f"[Broadcast] Model ID: {model_id}, Message: {message}")
            for connection in self.active_connections[model_id]:
                try:
                    client_info = getattr(connection, 'client', 'Unknown client')
                    print(f"  -> Sending to {client_info}")
                    await connection.send_text(message)
                except Exception as e:
                    print(f"  !! Error sending to {client_info}: {e}")

manager = ConnectionManager()

# 添加WebSocket端点
@app.websocket("/ws/training/{model_id}")
async def websocket_endpoint(websocket: WebSocket, model_id: int):
    await manager.connect(websocket, model_id)
    try:
        # 发送连接成功消息
        await websocket.send_text(json.dumps({
            "message": "WebSocket连接已建立",
            "model_id": model_id
        }))
        
        # 保持连接，等待消息
        while True:
            try:
                # 等待客户端消息，但不处理
                data = await websocket.receive_text()
                # 可以在这里处理客户端发送的消息，如果需要的话
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        manager.disconnect(websocket, model_id)
    except Exception as e:
        print(f"WebSocket错误: {str(e)}")
        manager.disconnect(websocket, model_id)

## 已过时,诸多细节与当前版本不符
# # 异步函数，用于发送训练结果
# async def send_example_training_results(model_id: int, train_details: dict):
#     # 等待一段时间，模拟训练过程
#     await asyncio.sleep(10)
    
#     # 检查CSV文件是否存在
#     csv_path = "car/results.csv"
#     if not os.path.exists(csv_path):
#         print(f"错误: 训练结果文件不存在: {csv_path}")
#         return
    
#     # 读取CSV文件
#     try:
#         with open(csv_path, 'r', encoding='utf-8') as csvfile:
#             csv_reader = csv.DictReader(csvfile)
#             rows = list(csv_reader)
            
#             # 如果没有数据，发送空数据消息
#             if not rows:
#                 print(f"没有训练结果数据: {csv_path}")
#                 return
            
#             # 处理每一行数据
#             processed_rows = []
#             for row in rows:
#                 processed_row = {}
#                 for key, value in row.items():
#                     # 去除键名中的空格
#                     clean_key = key.strip()
#                     # 去除值中的空格
#                     value = value.strip()
#                     # 尝试转换为数值类型
#                     try:
#                         # 尝试转换为整数
#                         processed_row[clean_key] = int(value)
#                     except ValueError:
#                         try:
#                             # 如果整数转换失败，尝试转换为浮点数
#                             processed_row[clean_key] = float(value)
#                         except ValueError:
#                             # 如果都失败，保持原始字符串
#                             processed_row[clean_key] = value
#                 processed_rows.append(processed_row)
            
#             # 获取训练细节中的epochs字段
#             epochs = train_details.get('epochs', 1000)  # 默认为1000
            
#             # 每秒发送一行数据
#             for i, row in enumerate(processed_rows):
#                 # 检查训练是否被暂停
#                 if model_id in training_status and training_status[model_id] == "suspend":
#                     # 发送暂停状态
#                     await manager.broadcast_to_model(model_id, json.dumps({
#                         "message": "训练已暂停",
#                         "status": "suspend"
#                     }))
#                     # 等待恢复训练
#                     while model_id in training_status and training_status[model_id] == "suspend":
#                         await asyncio.sleep(1)
#                     # 如果训练被取消，则退出
#                     if model_id not in training_status:
#                         return
                
#                 # 确定当前状态
#                 if i < len(processed_rows) - 1:
#                     status = "training"  # 还有更多数据，仍在训练中
#                     # 广播消息给特定模型ID的客户端，包含数据
#                     message = json.dumps({
#                         "message": "训练结果数据",
#                         "status": status,
#                         "data": row,
#                         "model_id": model_id
#                     })
#                     await manager.broadcast_to_model(model_id, message)
#                 else:
#                     # 最后一行数据，判断是否提前停止
#                     current_epoch = int(row.get('epoch', 0))
#                     if current_epoch < epochs:
#                         status = "early_stop"  # 提前停止
#                     else:
#                         status = "finished"  # 正常完成
                    
#                     # 发送最后一条数据，包含状态和数据
#                     message = json.dumps({
#                         "message": "训练结果数据",
#                         "status": training,
#                         "data": row,
#                         "model_id": model_id
#                     })
#                     await manager.broadcast_to_model(model_id, message)
                    
#                     # 发送完成状态，不包含数据
#                     message = json.dumps({
#                         "message": "训练结果数据发送完成",
#                         "status": status,
#                         "model_id": model_id
#                     })
#                     await manager.broadcast_to_model(model_id, message)
                
#                 await asyncio.sleep(1)  # 每秒发送一次
            
#             # 清理训练状态
#             if model_id in training_status:
#                 del training_status[model_id]
            
#     except Exception as e:
#         print(f"发送训练结果时出错: {str(e)}")
#         # 清理训练状态
#         if model_id in training_status:
#             del training_status[model_id]


def websocket_publish(message):
    if main_event_loop:
        asyncio.run_coroutine_threadsafe(message_queue.put(message), main_event_loop)
    else:
        print("主事件循环未设置，无法发送消息。")

def on_train_epoch_end(model_id:int):
    def callback(trainer):
        print("进入训练后回调函数")
    
        # curr_epoch = trainer.epoch + 1
        # text = f"Epoch Number: {curr_epoch}/{trainer.epochs} finished"
        # print(text)
        # print("-" * 50)

        # instance_variables = vars(trainer)
        # print(instance_variables)
        
        try:
            loss_dict = trainer.label_loss_items(trainer.tloss, prefix="train")
            metrics = trainer.metrics if trainer.metrics else {}
            # print("trainer.metrics keys:", trainer.metrics.keys())
            
            data = {
                "epoch": trainer.epoch + 1,
                "train/box_loss": loss_dict.get('train/box_loss'),
                "train/dfl_loss": loss_dict.get('train/dfl_loss'),
                "train/cls_loss": loss_dict.get('train/cls_loss'),
                "metrics/precision": metrics.get('metrics/precision(B)'),
                "metrics/recall": metrics.get('metrics/recall(B)'),
                "metrics/mAP_0.5": metrics.get('metrics/mAP50(B)'),
                "metrics/mAP_0.5:0.95": metrics.get('metrics/mAP50-95(B)'),
                "val/box_loss": metrics.get('val/box_loss'),
                "val/dfl_loss": metrics.get('val/dfl_loss'),
                "val/cls_loss": metrics.get('val/cls_loss'),
            }
        
            # 构建消息
            message = json.dumps({
                "message": "训练结果数据",
                "status": "training",
                "data": data,
                "model_id": model_id
            })
            websocket_publish(message)
            
        except Exception as e:
            print(f"发送训练回调数据时出错: {str(e)}")
            
            
    return callback
    
        
# def on_train_end(trainer):
#     # 切换训练状态
#     model_id = 0

#     current_epoch = trainer.epoch + 1
#     max_epochs = trainer.epochs
    
#     # 如果是最后一个epoch，状态为finished
#     if current_epoch >= max_epochs:
#         status = "finished"
#         training_status[model_id] = "finished"
#         print(f"\n训练完成！总轮次: {current_epoch}/{max_epochs}")
#     # 检查是否早停（如果有early_stop标志或者通过其他条件判断）
#     elif current_epoch < max_epochs:
#         status = "early_stop"
#         training_status[model_id] = "early_stop"
#         print(f"\n训练提前终止！轮次: {current_epoch}/{max_epochs}")
    
#     # 计算整体训练时间
#     total_training_time = trainer.end_time - trainer.start_time
#     print(f"训练结束！模型ID: {model_id}")
#     print(f"总训练时间: {total_training_time}")
    
#     message = json.dumps({
#         "message": "训练完成",
#         "status": status,
#         "total_training_time": str(total_training_time),
#         "best_epoch": best_epoch,
#         "best_accuracy": best_accuracy,
#         "best_model_path": trainer.best
#     })
    
#     websocket_publish(message)

#     print(f"最佳训练批次模型地址:{trainer.best} 精度: {trainer.best_accuracy}")
    
#     if model_id in training_status:
#         del training_status[model_id]
    
        
async def train_example(model_id: int, train_details: dict):
    await asyncio.sleep(1)  # 等待前端建立websocket连接
    
    # 启动线程并传递参数
    threading.Thread(target=run_training_process, args=(model_id, train_details), daemon=True).start()

def run_training_process(model_id: int, train_details: dict):
    # 初始化模型
    model = YOLO("weights/yolov8/yolov8s.pt", task="detect")
    
    # 添加回调函数
    model.add_callback("on_train_epoch_end", on_train_epoch_end(model_id))
    # model.add_callback("on_train_end", on_train_end)
    
    # 更新训练状态
    training_status[model_id] = "training"
    
    # 直接使用train_details中的参数进行训练
    model.train(
        data=r"/home/cvrsg/rs_workspace/aihub/backend_simp/datasets/split_car_dataset/sar.yaml",
        **train_details
    )
    
@app.on_event("startup")
async def start_background_task():
    global main_event_loop
    main_event_loop = asyncio.get_event_loop()

    async def broadcaster():
        while True:
            msg = await message_queue.get()
            await manager.broadcast_to_model(msg)

    asyncio.create_task(broadcaster())