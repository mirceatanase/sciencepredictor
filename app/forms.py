from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class PredictForm(FlaskForm):
    science_text = StringField('science_text', validators=[DataRequired()])
    submit = SubmitField('Analyze input text')