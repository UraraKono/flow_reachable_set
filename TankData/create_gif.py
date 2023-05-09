from PIL import Image
png_count = 3270
files = []
for i in range(10,png_count,10):
    seq = str(i)
    file_names = '3d_vis_' + seq + '.png'
    files.append(file_names)

print(files)
frames = []
files
for i in files:
    new_frame = Image.open(i)
    frames.append(new_frame)

# Save into a GIF file that loops forever
frames[0].save('3d_vis.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=40, loop=0)