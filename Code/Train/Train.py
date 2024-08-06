import logging
import torch


def evaluate(model, test_dl, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_dl:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    average_loss = running_loss / len(test_dl)
    accuracy = 100 * correct / total

    return average_loss, accuracy

def learn(model, train_dl, test_dl, weight_path, num_epochs, optimizer, criterion, scheduler, rank, device):
    logging.info('-' * 8 + f"Device {rank} Start Training" + '-' * 8)
    for epoch in range(num_epochs):
        model.train()
        correct, total = 0, 0
        for i, data in enumerate(train_dl):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        scheduler.step()

        if rank == 0:
            accuracy = 100 * correct / total
            logging.info(f"[Epoch {epoch + 1}] Train Acc: {accuracy:.2f}%")
            test_loss, test_accuracy = evaluate(model, test_dl, criterion, device)
            logging.info(f"[Epoch {epoch + 1}] Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

            torch.save(model.module.state_dict(), weight_path)

    return model
