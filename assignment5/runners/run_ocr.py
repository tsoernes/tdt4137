from ocrrunner import OCRRunner
import sys

print("Arguments: path resize_to_size window_size stride probability_threshold")
path = str(sys.argv[1])
resize_to_size = int(sys.argv[2])
window_size = int(sys.argv[3])
stride = int(sys.argv[3])
probability_threshold = int(sys.argv[4])
print(path)

runner = OCRRunner()
runner.run_ocr(path, resize_size, window_size, stride, prob_threshold)
