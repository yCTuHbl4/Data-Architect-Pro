CREATE TABLE astr
(
  base_uid  int8 NOT NULL GENERATED ALWAYS AS IDENTITY UNIQUE,
  tradedate date NOT NULL,
  open      real NOT NULL,
  low       real NOT NULL,
  high      real NOT NULL,
  numtrades int  NOT NULL,
  close     real NOT NULL,
  waprice   real NOT NULL,
  PRIMARY KEY (base_uid)
);

COMMENT ON TABLE astr IS 'Таблица цен ASTR';

COMMENT ON COLUMN astr.base_uid IS 'Индификатор записи';

COMMENT ON COLUMN astr.tradedate IS 'Дата торгов';

COMMENT ON COLUMN astr.open IS 'Цена открытия';

COMMENT ON COLUMN astr.low IS 'Наименьшия цена';

COMMENT ON COLUMN astr.high IS 'Наибольшая  цена';

COMMENT ON COLUMN astr.numtrades IS 'Количество сделок';

COMMENT ON COLUMN astr.close IS 'Цена закрытия';

COMMENT ON COLUMN astr.waprice IS 'Средневзвешенная цена';