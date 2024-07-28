import torch
import logging

def learn(model, dataloader, weight_path, num_epochs, optimizer, criterion, device):
    for epoch in range(num_epochs):
        running_loss = 0.0
        logging.info(f"Epoch: {epoch + 1}")
        for i, data in enumerate(dataloader):
            logging.info(f"Epoch: {epoch + 1}, step: {i + 1}")
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                logging.info(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0
        torch.save(model.state_dict(), weight_path)

    return model