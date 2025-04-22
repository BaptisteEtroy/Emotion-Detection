from model import predict_emotions
import torch.serialization
import numpy

# For PyTorch 2.6+ compatibility
torch.serialization.add_safe_globals([numpy._core.multiarray._reconstruct])

# Test emotion detection
test_text = "I'm excited about this project but also nervous about the deadline."

emotions = predict_emotions(test_text)

print(f"\nText: {test_text}")
print("\nDetected emotions:")
for emotion, data in emotions.items():
    if data['detected']:
        print(f"- {emotion}: {data['probability']:.3f}")

print("\nTop 3 emotions:")
for emotion, data in list(emotions.items())[:3]:
    print(f"- {emotion}: {data['probability']:.3f}")