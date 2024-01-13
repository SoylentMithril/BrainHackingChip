class Hackingchip:
    def __init__(self, ui_settings, settings, prompts):
        self.ui_settings = ui_settings
        self.settings = settings
        self.prompts = prompts
        
class HackingchipPrompts:
    def __init__(self, prompts, numpos, numneg):
        self.batch_prompts = prompts
        self.numpos = numpos
        self.numneg = numneg
        self.negend = numpos + numneg
        self.batch_size = numpos + numneg
        
