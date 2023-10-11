import cv2
import torch
import torchvision.transforms as transforms
import pythorch_my_model
from hand_dedect_live import Hand_dedector
model = pythorch_my_model.model
#pythorch_train_model de eğittiğim modeli yükledim.
model.load_state_dict(torch.load('pythorchTrainedModel.pth'))
model.eval()

# Open the camera
cap = Hand_dedector()
result_list = ['left_0', 'left_1', 'left_2', 'left_3', 'left_4', 'left_5',
               'right_0', 'right_1', 'right_2', 'right_3', 'right_4', 'right_5']
# Process frames in real-time
while True:
    # hand dedectet and normal camera frame
    frame, not_cropped_image = cap.get_cropped_frame()


    # Preprocess the image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    image = transform(frame)
    image = image.to(device="cuda")

    # image.to()
    # Make a prediction
    with torch.no_grad():
        output = model(image.unsqueeze(0)).to("cuda")
        _, prediction = torch.max(output, 1)
    print(prediction.item())
    # Display the result
    cv2.putText(frame, f'{result_list[prediction.item()]}',
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Finger Count', frame)
    cv2.imshow("Default Frame ", not_cropped_image)

    # Exit on ESC
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
