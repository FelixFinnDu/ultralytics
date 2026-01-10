# 添加ufnonet模块后的训练脚本
from ultralytics import YOLO


# 1. 并非加载预训练模型（这里用 YOLO11n，也可以换成 yolov11s、best.pt 等）
#    而是让YAML 定义自己的网络结构
# model = YOLO(model="/home/felix/workspace/ultralytics/ultralytics/cfg/models/11/yolo11n-ufnonet.yaml")
# model = YOLO(model="/home/felix/workspace/ultralytics/ultralytics/cfg/models/11/yolo11n-afnonet.yaml").load("yolo11n.pt")
model = YOLO("/home/felix/workspace/ultralytics/ultralytics/cfg/models/11/yolo11n.yaml")
# print(model.info())
# model.predict(source="https://ultralytics.com/images/bus.jpg", save=True)  # 测试预测
# 2. 开始训练
model.train(
    data="../datasets/ue5_yolo_dataset/data.yaml",  # 数据集配置文件
    # pretrained="/home/flash/workspace/vision/ultralytics/examples/ue_drone/tests/yolo11n.pt",  # 预训练权重
    cache=False,        # 是否缓存数据集到内存以加快训练速度
    epochs=200,        # 训练轮数（建议 100+）
    imgsz=640,         # 输入图片大小
    batch=32,          # batch size
    amp=True,          # 是否使用混合精度训练
    exist_ok=False,     # 覆盖已存在的结果
    project='runs/train',  # 结果保存路径
    name='ue5_drone_yolo11n',
    # verbose=True       # 打印详细训练日志
)

# 3. 训练完成后验证模型
# metrics = model.val()
# print("验证结果：", metrics)

# 4. 可选：导出为 ONNX/TorchScript 等格式
# model.export(format="onnx")
