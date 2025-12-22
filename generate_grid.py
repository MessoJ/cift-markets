from PIL import Image, ImageDraw

def create_grid_image(filename, size=(100, 100), color=(255, 255, 255, 0), line_color=(50, 50, 50, 50), step=20):
    image = Image.new('RGBA', size, color)
    draw = ImageDraw.Draw(image)
    
    for x in range(0, size[0], step):
        draw.line((x, 0, x, size[1]), fill=line_color)
        
    for y in range(0, size[1], step):
        draw.line((0, y, size[0], y), fill=line_color)
        
    image.save(filename)

if __name__ == "__main__":
    create_grid_image("c:/Users/mesof/cift-markets/frontend/public/grid.png")
