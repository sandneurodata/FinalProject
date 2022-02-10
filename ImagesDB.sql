CREATE DATABASE images;
USE images;

CREATE TABLE images_info(
	id VARCHAR(255),
    annotation VARCHAR (8000),
    width INT,
    height INT,
    cell_type VARCHAR(255)
);

DROP TABLE images_info;

SELECT * FROM images_info;

