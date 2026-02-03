import json

class RaidVisualizerExporter:
    def __init__(self):
        self.segments = []

    def addSegment(self, offset, raw, prediction, isCorrect, trueLabel, metadata):

        content = raw.hex() #json pt si bin

        seg = {
            "id": f"seg_{offset}",
            "offset": offset,
            "size": len(raw),
            "prediction": prediction,
            "isCorrect": isCorrect,
            "trueLabel": trueLabel,
            "contentHex": content,
            "metadata": metadata or {}
        }

        self.segments.append(seg)

    def saveJson(self, filepath="raid_visualization.json"):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.segments, f, indent=4)
            print(f"Fichier de visualisation sauvegardé : {filepath}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du fichier de visualisation : {e}")