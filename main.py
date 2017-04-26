from ImageInput import ImageInput
import sys

if __name__ == '__main__':
 
    # creates instance of class and loads image    
    image_input = ImageInput('TestImage/Machine.jpg')
    # plots preprocessed imae 
    image_input.plot_preprocessed_image()
    # detects objects in preprocessed image
    candidates = image_input.get_text_processed_image()
    # plots objects detected
    image_input.plot_to_check(candidates, 'Total Objects Detected')
    # selects objects containing text
    maybe_text = image_input.select_text_among_candidates('linearsvc-train-textvsothers.pickle')
    # plots objects after text detection
    image_input.plot_to_check(maybe_text, 'Objects Containing Text Detected')
    # classifies single characters
    classified = image_input.classify_text('linearsvc-train-textvstext.pickle')
    # plots letters after classification 
    image_input.plot_to_check(classified, 'Character Recognition')
    # plots the realigned text
    image_input.realign_text()
    
##########################################################################################################################    
## MACHINE LEARNING SECTION
##########################################################################################################################    
#    from data import OcrData
#    from cifar import Cifar
#    
#    ###################################################################
#    # 1- GENERATE MODEL TO PREDICT WHETHER AN OBJECT CONTAINS TEXT OR NOT
#    ###################################################################
#    
#    #CREATES AN INSTANCE OF THE CLASS LOADING THE OCR DATA 
#    data = OcrData('configOcr.py')
#    
#    #GENERATES A UNIQUE DATA SET MERGING NON-TEXT WITH TEXT IMAGES
#    data.merge_cifar()
#    
#    #PERFORMS GRID SEARCH CROSS VALIDATION GETTING BEST MODEL OUT OF PASSED PARAMETERS
#    data.perform_grid_search_cv('linearsvc-hog')
#    
#    #TAKES THE PARAMETERS LINKED TO BEST MODEL AND RE-TRAINS THE MODEL ON THE WHOLE TRAIN SET
#    data.generate_best_hog_model()
#    
#    #TAKES THE JUST GENERATED MODEL AND EVALUATES IT ON TRAIN SET
#    data.evaluate('linearsvc-train-textvsothers.pickle')
#
#
#    ###################################################################
#    # 2- GENERATE MODEL TO CLASSIFY SINGLE CHARACTERS
#    ###################################################################
#    
#    #CREATES AN INSTANCE OF THE CLASS LOADING THE OCR DATA 
#    #data = OcrData('configOcr.py')
#    
#    #PERFORMS GRID SEARCH CROSS VALIDATION GETTING BEST MODEL OUT OF PASSED PARAMETERS
#    #data.perform_grid_search_cv('linearsvc-hog')
#    
#    #TAKES THE PARAMETERS LINKED TO BEST MODEL AND RE-TRAINS THE MODEL ON THE WHOLE TRAIN SET
#    #data.generate_best_hog_model()
#    
#    #TAKES THE JUST GENERATED MODEL AND EVALUATES IT ON TRAIN SET
#    #data.evaluate('linearsvc-train-textvstext.pickle')
