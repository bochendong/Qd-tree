import logging
import torch

def learn(model, dataloader, weight_path, num_epochs, optimizer, criterion, scheduler, rank, device):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 100 == 99:
                logging.info(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        accuracy = 100 * correct / total
        logging.info(f"[Epoch {epoch + 1}] Accuracy: {accuracy:.2f}%")

        scheduler.step()

        if rank == 0 and (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), weight_path)

    return model
