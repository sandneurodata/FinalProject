import streamlit as st
from PIL import Image
from helpers_v2 import *
from numpy import *
import numpy as np
import pandas as pd
import sqlalchemy

# -- Get the data necessary
current_dir = '/Users/mitochondria/Documents/Codingschool/DataScienceCourse/Final_Project/sartorius-cell-instance-segmentation/'
img_folder = '/Users/mitochondria/Documents/Codingschool/DataScienceCourse/Final_Project/sartorius-cell-instance-segmentation/train/'
plot_folder = '/Users/mitochondria/Documents/Codingschool/DataScienceCourse/Final_Project/sartorius-cell-instance-segmentation/plot/'
train_csv = pd.read_csv('/Users/mitochondria/Documents/Codingschool/DataScienceCourse/Final_Project/sartorius-cell-instance-segmentation/train.csv', sep=',')

img_id_list = train_csv['id'].unique().tolist()

# -- Create tmp dataframe that will hold the info for the newly segmented images
tmp_df = pd.DataFrame(columns = ['id', 'annotation','width','height','cell_type'])

# -- Config the page
app_title = 'Img Seg'
st.set_page_config(page_title = app_title, page_icon = ":national_park:", layout = 'wide', initial_sidebar_state = 'collapsed')


# -- Title of the app and explanation
st.title('Cell Segmentation App')
with st.expander('App Info'):
    st.markdown(""" 
        * Display images and mask of the cells
        * Save the mask onto a predefined database
        * To start using the app, click the arrow on the left hand side to select the image folder and input settings
      """)
    
# -- Sidebar building
with st.sidebar:
    st.header('Image Settings & Segment')
    with st.expander('Load and Display Img'):
        sub_dir = st.selectbox("Select Directory", getSubDir(current_dir), index=0)
        #file_names = st.file_uploader("Load Img File(s)", type=("png", "jpeg", "jpg","tiff"), accept_multiple_files = True) 
        img_format = st.selectbox("Select Img Format", img_format, index=0)
        img_list = getImgList(sub_dir, img_format)
        if len(img_list) > 0:
            img_name = st.selectbox("Select Image #", img_list, index = 0)
            st.write(img_list[0].split('/')[-1])
#         if len(img_list) == 2:
#             img_name = st.selectbox("Select Image #", img_list, index = 1)
#         else:
#             img_name = st.selectbox("Select Image #", img_list, index = 0)
#         seg_check = st.checkbox("Run Segmentation")
    
    st.header('Database Settings')
    db_expander = st.expander('Database Info')
    with db_expander:
        db_name = st.text_input('Database Name', 'images')
        db_table = st.text_input('Table Name', 'images_info')
        db_host = st.text_input('Database Host', '127.0.0.1')
        db_user = st.text_input('Database User', 'root')
        db_port = st.text_input('Database Port', '3306')
        db_password = st.text_input("Database Password", 'Sand250880BioCEL65!!', type="password")
        db_save = st.checkbox('Save to Database')
        db_fetch = st.checkbox('Fetch from Database')
      

con = f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{int(db_port)}/{db_name}' 
db_data = pd.read_sql(
    sql = f"""
    SELECT * FROM {db_table} 
    """, 
    con = con)

# -- Display images and results from image segmentation:
if img_name != " ":
    id_img = img_name.split('.')[0]
    if id_img in img_id_list:
        col1, col2, col3 = st.columns(3)
#         org_img = loadImg(id_img, img_format, sub_dir, 'org')
#         true_mask = loadImg(id_img, img_format, sub_dir, 'true')
#         pred_mask = loadImg(id_img, img_format, sub_dir, 'pred')
        with col1:
            fig = getFig(current_dir+sub_dir+'/'+id_img+'.'+img_format)
            st.pyplot(fig)
            st.caption(id_img+'.'+img_format)
        with col2:
            fig = getFigMask(current_dir, sub_dir, id_img, img_format)
            st.pyplot(fig)
            st.caption(id_img+'.'+img_format +' and true mask')
        with col3:
            fig = getFigPred(current_dir, sub_dir, id_img, img_format)
            st.pyplot(fig)
            st.caption(id_img+'.'+img_format+' and prediction mask')
    else:
        col1, col2 = st.columns(2)
        pred_df = getDataFrame(current_dir, sub_dir, id_img, img_format)
        with col1:
            fig = getFig(current_dir+sub_dir+'/'+id_img+'.'+img_format)
            st.pyplot(fig)
            st.caption(id_img+'.'+img_format)
        with col2: 
            fig = getFigPred(current_dir, sub_dir, id_img, img_format)
            st.pyplot(fig)
            st.caption(id_img+'.'+img_format+' and prediction mask')
        
          
            
with st.expander('Segmentation Results'):
    if img_name != " ":
        st.write(f'Results for {id_img}:')
        if id_img in train_csv['id'].unique():
            st.dataframe(train_csv.query('id == @id_img').reset_index().filter(['id','annotation','width','height','cell_type']))
        else:
            st.dataframe(pred_df)

            
with st.expander('Database Data'):
    if db_fetch:
        st.dataframe(db_data)
    
# -- Check if database exists and update it if necessary        
if db_save:
    if id_img in train_csv['id'].unique():
        if id_img not in db_data['id'].unique():
            tmp = train_csv.query('id == @id_img').reset_index().filter(['id','annotation','width','height','cell_type'])
            tmp.to_sql(db_table, if_exists='append', con=con, index=False)
    else:
        if id_img not in db_data['id'].unique():
            pred_df.to_sql(db_table, if_exists='append', con=con, index=False)
        
            
            
#     if id_img not in db_data['id'].unique():
#         if id_img in train_csv['id'].unique():
#             tmp = train_csv.query('id == @id_img').reset_index().filter(['id','annotation','width','height','cell_type'])
#             tmp.to_sql(db_table, if_exists='append', con=con, index=False)
#     else:
#         pred_df.to_sql(db_table, if_exists='append', con=con, index=False)