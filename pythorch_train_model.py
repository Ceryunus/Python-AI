from tqdm.notebook import tqdm
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import os

# [Tr] Veri yüklemek için gerekli modül [En] Required module for loading data
import pythorch_data_load
# [Tr] Model, loss ve optimizer'ın tanımlandığı modül [En] Module where model, loss and optimizer are defined
import pythorch_my_model


# [Tr] Hata mesajını önlemek için gerekli satır [En] Required line to prevent error message
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
train_data = pythorch_data_load.train_data
test_data = pythorch_data_load.test_data

# [Tr] Model tanımlanıyor [En] Defining the model
model = pythorch_my_model.model
# [Tr] Loss fonksiyonu tanımlanıyor [En] Defining the loss function
criterion = pythorch_my_model.criterion
# [Tr] Optimizer tanımlanıyor [En] Defining the optimizer
optimizer = pythorch_my_model.optimizer
size = len(train_data.dataset)

# [Tr] Epoch sayısı belirleniyor [En] Number of epochs is determined
n_epoch = 5
# [Tr] GPU belleği temizleniyor [En] Clearing the GPU memory
torch.cuda.empty_cache()
# [Tr] Model eğitim moduna alınıyor [En] Setting the model to training mode
model = model.train()

# [Tr] Her bir epoch için [En] For each epoch
for epoch in range(n_epoch):
    epoch_loss = 0
    epoch_acc = 0

    for i, (x, y) in tqdm(enumerate(train_data)):
        y = y.reshape(-1)

        predict = model(x)

        # [Tr] Loss hesaplanıyor CrossEntropy ile  [En] Calculating the loss with CrossEntropy
        loss = criterion(predict, y)
        epoch_loss += loss / len(train_data)
        correct_prediction = torch.argmax(predict, 1) == y
        correct_prediction = correct_prediction.sum()
        epoch_acc += correct_prediction

        # [Tr] Gradyan sıfırlanıyor [En] Zeroing the gradients
        optimizer.zero_grad()

        loss.backward()

        # [Tr] Gradyanlar kullanılarak ağırlıklar güncelleniyor [En] Updating the weights using gradients
        optimizer.step()

        # [Tr] Her 100 örnekte [En] Every 100 examples
        if i % 100 == 0:
            loss, current = loss.item(), (i+1)*len(x)
            # [Tr] Loss değeri ve örnek sayısı yazdırılıyor [En] Printing the loss value and number of examples
            print(f"loss:{loss} current :{current}/{size}")

    # [Tr] Epoch sonunda toplam loss ve doğruluk değerleri yazdırılıyor [En] Printing the total loss and accuracy values at the end of each epoch
    epoch_acc = epoch_acc / (32 * len(train_data))
    print('Epoch : {}/{},   loss : {:.5f},    acc : {:.5f}'.format(epoch +
          1, n_epoch, epoch_loss, epoch_acc))

    # [Tr] Doğruluk değeri belirli bir eşiği geçerse döngü sonlandırılıyor [En] Exiting the loop if the accuracy value exceeds a certain threshold
    if epoch_acc > 0.08:
        break

# [Tr] modeli test etmek için. Modeli eğitmek ile benzer. [En] to test the model. Similar with training the model.
with torch.no_grad():
    val_loss = 0
    val_acc = 0
    result_list = ['left_0', 'left_1', 'left_2', 'left_3', 'left_4', 'left_5',
                   'right_0', 'right_1', 'right_2', 'right_3', 'right_4', 'right_5']

    for i, (x, y) in enumerate(test_data):
        y = y.reshape(-1)

        predict = model(x)
        loss = criterion(predict, y)

        val_loss += loss / len(test_data)
        correct_prediction = torch.argmax(
            predict, 1) == y
        correct_prediction = correct_prediction.sum()
        val_acc += correct_prediction

        if i < 5:
            r = random.sample(range(0, len(x)), 5)
            fig = plt.figure(figsize=(22, 22))
            # prediction
            # 0 1 2 3 4 5 6 7 8 9 10 11

            for i in range(5):
                label = predict[r[i]]
                label = torch.argmax(label).item()

                img = x[r[i]].to('cpu').numpy()
                img = np.transpose(img, (1, 2, 0))

                subplot = fig.add_subplot(1, 5, i+1)
                subplot.set_title(result_list[label])
                subplot.imshow(img, cmap=plt.cm.gray_r)
            plt.show()

    val_acc = val_acc.item() / (32 * len(test_data))
    print('loss : {:.5f},    acc : {:.5f}'.format(val_loss, val_acc))

# [Tr] Eğitilmiş Model kaydediliyor [En] Saving the trained model
PATH = "./AdamaxS4.pth"
torch.save(model.state_dict(), PATH)
print("model saved: ", PATH)
