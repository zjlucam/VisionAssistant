import matplotlib.pyplot as plt
from textwrap import wrap

def show_image_captions(model, dataloader, processor, device, num_samples=40):
    model.eval()
    rows = (num_samples + 4) // 5
    fig, axes = plt.subplots(rows, 5, figsize=(20, rows * 4))
    axes = axes.flatten()

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            generated_ids = model.generate(pixel_values=pixel_values, max_new_tokens=77)
            captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
            ground_truths = processor.batch_decode(input_ids, skip_special_tokens=True)

            images = pixel_values.permute(0, 2, 3, 1).cpu().numpy()

            for ax, img, cap, gt in zip(axes, images, captions, ground_truths):
                ax.imshow(img)
                ax.set_title("\n".join(wrap(f"Generated: {cap}\nActual: {gt}", width=40)), fontsize=8)
                ax.axis('off')

    plt.tight_layout()
    plt.show()
