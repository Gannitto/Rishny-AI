import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
input_details = session.get_inputs()[0]
print("Имя входного тензора:", input_details.name)
print("Ожидаемая форма:", input_details.shape)
print("Тип данных:", input_details.type)