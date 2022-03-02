# mosaicImage
Turns an image into a mosaic version with a limited selection of colours

Family members of mine requested some ideas for abstract art. They wanted to use pieces of felt as acoustic insulation, but wanted it to look nice. 
I had an idea and developed a program that would not only pixelate a given image, but reduce its colours to only n few dominant ones. This was necessary due to fabric (and especially felt) coming in a limited colour range. 
Selecting the n dominant colours was achieved via kmeans clustering, as normal methods to derive dominant colours were prone to miss colours that were rare but vital to an image (like the tiny yellow beak of a bird). 
Every colour of the pixelated version of the original image is then replaced by the dominant colour matching it most closely. 
My program produces additional outputs: An overview-colour palette with the percentages at which a given colour is occurring in the final image, and a table showing how many “pixels” are coloured in each of the n final colours. The latter can now be used to buy the right amount of fabric per dominant colour.