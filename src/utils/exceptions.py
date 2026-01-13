class FraudException(Exception):
      # REMARK: [PROJECT-SPECIFIC] – class ka naam project ke hisaab se hai


  def __init__(self, message, error):
    super().__init__(message)
    self.error = error


  def __str__(self):
    return f"{self.args[0]} | Root cause: {self.error}"