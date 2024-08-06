import logging
import torch

def learn(model, dataloader, weight_path, num_epochs, optimizer, criterion, scheduler, rank, device):
    logging.info('-' * 8 + f"Device {rank} Start Training" + '-' * 8)
    for epoch in range(num_epochs):
        model.train()
        # dataloader.sampler.set_epoch(epoch)
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            logging.info(f"Device {rank} Batch {i + 1}: inputs size: {inputs.size()}, labels size: {labels.size()}")
            inputs, labels = inputs.to(device), labels.to(device)
            logging.info(f"Device {rank} Batch {i + 1}: inputs min: {inputs.min()}, inputs max: {inputs.max()}")

            '''
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (rank == 0):
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # logging.info(f"Device {rank} [Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}")
                running_loss = 0.0'''

        '''if (rank == 0):
            accuracy = 100 * correct / total
            logging.info(f"[Epoch {epoch + 1}] Acc: {accuracy:.2f}%")
        '''
        scheduler.step()

        if rank == 0 and (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), weight_path)

    return model
