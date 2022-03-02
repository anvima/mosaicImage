# mosaicImage
*Turns an image into a mosaic version with a limited selection of colours*

Family members of mine requested some ideas for abstract art which would double as acoustic insulation. Their only prerequisite was to use colourful felt/fabric on the piece. 

I had an idea and developed a small program that would not only pixelate a given image, but reduce its colours to only the n dominant ones. This was necessary due to fabric (and especially felt) coming in a limited colour range. 

Selecting the n dominant colours was achieved via kmeans clustering, as normal methods to derive dominant colours were prone to miss colours that were rare but vital to an image (like the tiny yellow beak of a bird). 

Every colour of the pixelated version of the original image is then replaced by the dominant colour matching it most closely. 

My program produces additional outputs: An overview-colour palette with the percentages at which a given colour is occurring in the final image, and a table showing how many “pixels” are coloured in each of the n final colours. The latter can now be used to buy the right amount of fabric per dominant colour.

Original image           | Pixelated image with simplified colours      |  Colour palette 
:-----------------------:|:----------------------------------------------:|:---------------:
<img src="https://user-images.githubusercontent.com/53763279/156440543-2a3710ce-6467-4b5a-8daf-d7da86e2e4c4.JPG" height="300" > | <img src="https://user-images.githubusercontent.com/53763279/156440552-49313705-84f2-46ba-bf6f-bd62197e6fce.jpg" height="300" > | <img src="https://user-images.githubusercontent.com/53763279/156440567-afb4a6e3-293a-408e-b319-e2f7a13ac621.jpg" height="300" >

User options: 
- input image (title)
- amount of dominant colours the resulting image should consist of 
- should the constrast of the input image be increased? (advisable for most images)
- scale factor: How large should the resulting pixels be? High scale factor = higher level of abstraction (fewer/larger pixels). 

Note that the amount of final colours the resulting image consists of may potentially be larger than the amount selected. This may happen due to a feature of the program. In general, for selecting the n dominant colours, the program applies kmeans clustering and picks one colour per cluster (the centroid). Through the implemented feature, it then additionally it looks at the sizes of the individual clusters. If a cluster is very small (holds < 3% of instances), the program selectes more than one colour for this cluster. It does this to produce depth in small colour areas: For example, if there is just one small area of an image that consists of shades of blue, "blue" would most probably form just one cluster -> one shade of blue in the final image. Consequently, the one small blue area would look flat. The program circumvents this issue with the additional colour selection. Example: 

Snippet of original image           | Snippet of pixelated image with simplified colours     
:----------------------------------:|:-----------------------------------------------------:
<img src="https://user-images.githubusercontent.com/53763279/156447009-59f63d87-94e5-4769-9746-95563942e5c5.png" height="100" > | <img src="https://user-images.githubusercontent.com/53763279/156447043-f1dc30b9-ef6f-458a-a180-4eff7b00a0d6.png" height="100" > 

This snippet is part of a larger image with only this small blue area. Without the small cluster-feature, just one shade of blue (and one shade of yellow) would have been part of the final result. 
