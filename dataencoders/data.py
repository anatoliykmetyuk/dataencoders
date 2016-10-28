import numpy as np

CHAR_TO_ID = 'charToId'
ID_TO_CHAR = 'idToChar'

class CharEnc:

  def __init__(self, charMap, sequenceLength):
    self.charMap        = charMap
    self.sequenceLength = sequenceLength

    self.charToId       = charMap[CHAR_TO_ID]
    self.idToChar       = charMap[ID_TO_CHAR]
    self.embeddingLen   = len(self.charToId)

  def addChar(self, char, id):
    newCTI = dict(self.charToId)
    newITC = dict(self.idToChar)

    newCTI.update({char: id})
    newITC.update({id: char})

    newCharMap = {CHAR_TO_ID: newCTI, ID_TO_CHAR: newITC}
    return CharEnc(newCharMap, self.sequenceLength)

  # Conversions between list[string] and array2D[np.int]

  def textToId(self, txts):
    res = np.zeros((len(txts), self.sequenceLength), dtype = np.int)
    for i, txt in enumerate(txts):
      for j, char in enumerate(txt):
        if j < self.sequenceLength and char in self.charToId:
          res[i, j] = self.charToId[char]
    return res

  def idToText(self, ids):
    res = []
    for sample in ids:
      txt = ""
      for charId in sample: txt = txt + self.idToChar[charId]
      res.append(txt)
    return res


  # Conversions between array2D[np.int] and one-hot array3D[np.int]

  def idToOneHot(self, ids):
    res = np.zeros(ids.shape + (self.embeddingLen,), dtype=np.int)
    for i in range(ids.shape[0]):
      for j in range(ids.shape[1]):
        res[i, j, ids[i, j]] = 1
    return res

  def oneHotToId(self, oneHot):
    res = np.zeros(oneHot.shape[:-1], dtype=np.int)
    for i in range(oneHot.shape[0]):
      for j in range(oneHot.shape[1]):
        vec = oneHot[i,j]
        res[i, j] = np.argmax(vec)
    return res


  # Direct conversions between one-hot and text

  def textToOneHot(self, text): return self.idToOneHot(self.textToId(text))
  def oneHotToText(self, oh  ): return self.idToText(self.oneHotToId(oh))

def charMap(text):
  chars = sorted(list(set(text)))
  charToId = dict((c, i) for i, c in enumerate(chars))
  idToChar = dict(enumerate(chars))
  return {'charToId': charToId, 'idToChar': idToChar}

def charMapList(texts): return charMap("".join([row for row in texts]))
