import json

class RaidVisualizerExporter:
    def __init__(self):
        self.segments = []

    def addSegment(self, offset, size, prediction, isCorrect, trueLabel, metadata=None):

        seg = {
            "o": offset, #ne pas changer les clés, optimisation taille fichier
            "s": size,
            "p": prediction,
            "c": isCorrect,
            "t": trueLabel,
            "m": metadata or {}
        }

        self.segments.append(seg)

    def saveJson(self, filepath="raid_visualization.json"):
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.segments, f, indent=4)
            print(f"Fichier de visualisation sauvegardé : {filepath}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du fichier de visualisation : {e}")