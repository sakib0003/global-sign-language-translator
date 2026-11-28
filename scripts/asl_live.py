import cv2
import torch
import mediapipe as mp
import numpy as np
import torch.nn as nn

MODEL_PATH = "models/asl_alphabet_cnn.pth"
DEVICE = torch.device("cpu")

class ASLCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
classes = checkpoint["classes"]
model = ASLCNN(len(classes)).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Webcam could not be opened")

print("ASL Live Recognition Started")
print("Press Q to quit")

prediction = "Show your hand"
confidence = 0.0

while True:
    ret, frame = camera.read()
    if not ret:
        print("Camera frame failed")
        break

    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        points = result.multi_hand_landmarks[0].landmark
        xs = [int(p.x * width) for p in points]
        ys = [int(p.y * height) for p in points]

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        pad = int(max(x2-x1, y2-y1) * 0.35)
        cx = (x1+x2)//2
        cy = (y1+y2)//2
        side = max(x2-x1, y2-y1) + 2*pad

        left = max(0, cx-side//2)
        top = max(0, cy-side//2)
        right = min(width, cx+side//2)
        bottom = min(height, cy+side//2)

        crop = frame[top:bottom, left:right]

        if crop.size > 0:
            image = cv2.resize(crop, (64, 64))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(
                np.ascontiguousarray(image)
            ).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0)

            with torch.no_grad():
                output = model(tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, index = probabilities.max(dim=1)
                prediction = classes[index.item()]
                confidence = confidence.item() * 100

            cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
    else:
        prediction = "No hand detected"
        confidence = 0.0

    cv2.putText(
        frame, f"ASL: {prediction}",
        (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0,255,0), 2
    )
    cv2.putText(
        frame, f"Confidence: {confidence:.1f}%",
        (20, 85), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0,255,0), 2
    )

    cv2.imshow("Global Sign Language Translator - ASL", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
hands.close()
cv2.destroyAllWindows()
print("ASL Webcam stopped")
