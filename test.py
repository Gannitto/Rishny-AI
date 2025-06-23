import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
print(session.get_inputs()[0])