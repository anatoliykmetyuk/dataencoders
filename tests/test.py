import numpy as np
from dataencoders import CharEnc, charMap, charMapList

def debug(what, msg):
  print(msg, "\n", what, "\n")

def test():
  np.set_printoptions(threshold=np.nan)

  originalText = ["Foo bar"]
  converter    = CharEnc(charMapList(originalText), 20)
  debug(converter.charMap, "Char map")
  debug(originalText, "Original text")

  idMat        = converter.textToId  (originalText)
  debug(idMat, "Id-encoded text")

  oneHotMat    = converter.idToOneHot(idMat)
  debug(oneHotMat, "One-hot encoded text")

  idMat2       = converter.oneHotToId(oneHotMat)
  debug(idMat2, "Id-encoded text, obtained by decoding the one-hot encoded text")
  
  txt          = converter.idToText  (idMat)
  debug(txt, "String text, obtained by decoding the id-encoded text")

test()