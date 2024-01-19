from imports import *

path = ("https://raw.githubusercontent.com/remijul/dataset/master/SMSSpamCollection")
df = pd.read_csv(path, sep="\t", names = ['Flag','Text'])

st.write("Ma data frame")
st.dataframe(df)

df['Longeur du document'] = df['Text'].str.split(r"(?!^)").str.len()
df['Nombres de mots'] = df['Text'].str.split().str.len()

st.dataframe(df)

# st.sidebar → colonne de gauche
st.sidebar.title('Configuration')
nl = st.sidebar.slider('Lignes',
min_value=0,
max_value=min(50,df.shape[0]))
# partie centrale
st.table(df.iloc[0:nl])

st.button('Hit me')
st.checkbox('Check me out')
st.sidebar.radio('Pick one:', ['spam','ham','none'])
st.selectbox('Select', [1,2,3])
st.multiselect('Multiselect', [1,2,3])
st.slider('Slide me', min_value=0, max_value=10)
st.select_slider('Slide to select', options=[1,'2'])
st.text_input('Enter some text')
st.number_input('Enter a number')
st.text_area('Area for textual entry')
st.date_input('Date input')
st.time_input('Time entry')
st.file_uploader('File uploader')
st.color_picker('Pick a color')