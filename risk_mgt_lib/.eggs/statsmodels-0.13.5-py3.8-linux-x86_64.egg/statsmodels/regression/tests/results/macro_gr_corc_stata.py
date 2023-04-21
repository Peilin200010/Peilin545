import numpy as np

from statsmodels.tools.testing import ParamsTableTestBunch
est = dict(
    N=201,
    df_m=2,
    df_r=198,
    F=221.0377347263228,
    r2=.6906614775140222,
    rmse=10.66735221013527,
    mss=50304.8300537672,
    rss=22530.89582866539,
    r2_a=.6875368459737599,
    ll=-759.5001027340874,
    N_gaps=0,
    tol=1.00000000000e-06,
    max_ic=100,
    ic=4,
    dw=1.993977855026291,
    dw_0=2.213805016982909,
    rho=-.1080744185979703,
    rank=3,
    cmd="prais",
    title="Prais-Winsten AR(1) regression",
    cmdline="prais g_realinv g_realgdp L.realint, corc rhotype(tscorr)",
    tranmeth="corc",
    method="iterated",
    depvar="g_realinv",
    predict="prais_p",
    rhotype="tscorr",
    vce="ols",
    properties="b V",
)

params_table = np.array([
    4.3704012379033,
    .20815070994319,
    20.996331163589,
    2.939551581e-52,
    3.9599243998713,
    4.7808780759353,
    198,
    1.9720174778363,
    0,
    -.5792713864578,
    .26801792119756,
    -2.1613158697355,
    .03187117882819,
    -1.1078074114328,
    -.05073536148285,
    198,
    1.9720174778363,
    0,
    -9.509886614971,
    .99049648344574,
    -9.6011311235432,
    3.656321106e-18,
    -11.463162992061,
    -7.5566102378806,
    198,
    1.9720174778363,
    0]).reshape(3, 9)

params_table_colnames = 'b se t pvalue ll ul df crit eform'.split()

params_table_rownames = 'g_realgdp L.realint _cons'.split()

cov = np.array([
    .04609356125016,
    -.00228616599156,
    -.13992917065996,
    -.00228616599156,
    .08103590074551,
    -.10312637237487,
    -.13992917065996,
    -.10312637237487,
    1.1416832888557]).reshape(3, 3)

cov_colnames = 'g_realgdp L.realint _cons'.split()

cov_rownames = 'g_realgdp L.realint _cons'.split()

fittedvalues = np.array([
    34.092961143383,
    -12.024019193439,
    -4.0322884907855,
    26.930266763436,
    -18.388570551209,
    -8.1324920108371,
    -31.875322937642,
    .21814034941388,
    21.286872048596,
    18.046781817358,
    24.839977757054,
    20.524927286892,
    9.4135589847345,
    5.0452309081846,
    -5.6551802066401,
    11.985460837387,
    10.881000973745,
    22.932381778173,
    2.2243337561549,
    28.678018086877,
    8.4960232938048,
    12.595931232461,
    -5.9735838194583,
    31.918171668207,
    12.51486507374,
    24.858076725235,
    30.404510147102,
    32.036251557873,
    -3.4858432808681,
    .47271857077911,
    4.4052813399748,
    3.2797897853075,
    -10.188593201973,
    4.2830255779438,
    3.2910087767189,
    26.047109629787,
    18.955198396277,
    2.5715957164729,
    -2.3281798982217,
    17.032381944743,
    -4.0901785042795,
    .91987313367478,
    -18.683722252157,
    -12.984352504073,
    -6.6160771495293,
    4.5273942020195,
    -28.749156193466,
    38.148138299645,
    -.57670949517535,
    4.4996010868259,
    -5.6725761358388,
    20.923003763987,
    31.094336738501,
    6.5964070665898,
    18.667129468109,
    34.425102067117,
    12.495888351027,
    -20.391896794875,
    9.6404042884163,
    -23.337183583421,
    -3.2089713311258,
    -25.745423471535,
    -13.275374642526,
    -29.102203184727,
    3.6718187868943,
    20.796143264795,
    13.375486730466,
    30.503754833333,
    1.9832701043709,
    -.33573283324142,
    3.8454619057923,
    11.198841650909,
    27.300615912881,
    21.617271919636,
    -10.213387774549,
    -3.041115945304,
    58.662013692103,
    9.3875351174185,
    14.62429232726,
    -7.0289661733807,
    -6.3287417283568,
    5.3688078321969,
    -3.9071766792954,
    -2.3350903711443,
    -45.218293861742,
    -12.518622374718,
    22.354531985202,
    24.642689788549,
    -26.596828143888,
    8.8860015710942,
    -35.108835611339,
    -42.53255546273,
    -6.0981548538498,
    -17.177799997821,
    -11.40329039155,
    7.0180843697455,
    26.705585327159,
    21.928461900264,
    23.355257443029,
    21.894505946705,
    17.657379506229,
    3.4294676089343,
    .97440677130275,
    3.509217298751,
    3.2828913970637,
    14.944972085155,
    1.2923006012963,
    6.0435554866624,
    -8.8442213362579,
    5.4509233390479,
    -2.6794059247137,
    -.48161809935521,
    8.4201161664985,
    4.549532431433,
    18.996304264647,
    -1.8170413137623,
    11.849281819014,
    -1.7066851102854,
    12.218564198008,
    4.6715818362824,
    2.108627710813,
    2.1972303599907,
    -8.5249858114578,
    8.0728543937531,
    -4.5685185423019,
    -11.135138151837,
    -24.047910391406,
    -19.615146607087,
    -.44428684605231,
    -3.4877810606396,
    -3.9736701841582,
    9.0239662841874,
    8.5557600343168,
    8.2271736548708,
    9.0458357352972,
    -6.3116081663522,
    1.5636968490473,
    -.95723547143789,
    13.44821815082,
    6.7721586886424,
    13.649238514998,
    1.1579094135284,
    8.6396143031575,
    -6.7325794361447,
    -7.0767574403351,
    3.1400395218988,
    .91586368309611,
    1.2368141501238,
    19.694940608963,
    4.0145204498294,
    8.3107956501685,
    2.7428657521209,
    13.988676494758,
    10.116898229369,
    2.524027931198,
    4.6794023157157,
    3.5155627033849,
    11.945878601415,
    18.880925278224,
    4.5865622289454,
    3.2393307465763,
    11.062953867859,
    20.793966231154,
    -6.3076444941097,
    23.178864105791,
    -8.9873099715132,
    -1.107322743866,
    -16.332040882272,
    .42734142108626,
    -15.050781890016,
    -4.6612306163595,
    4.565652288529,
    .80725599873503,
    -.87706444767528,
    -8.5407936802022,
    -1.3521383036839,
    4.4765977604986,
    19.623863498831,
    7.1113956960438,
    3.9798487855641,
    3.6934842863203,
    4.6801104799091,
    6.7218162617593,
    7.7832579175778,
    -1.2290990957424,
    3.0474310004174,
    2.7567736850761,
    11.188206993423,
    -4.3306276498455,
    -9.5365114805844,
    -.53338170341178,
    -5.206342794124,
    4.1154674910376,
    4.7884361973806,
    -.64799653797949,
    -10.743852791188,
    -2.461403042047,
    -17.431541988995,
    -36.151189705211,
    -43.711601400093,
    -12.334881925913,
    4.3341943478598])

fittedvalues_colnames = 'fittedvalues'.split()

fittedvalues_rownames = ['r'+str(n) for n in range(1, 203)]

fittedvalues_se = np.array([
    1.6473872957314,
    1.0113850707964,
    .7652190209006,
    1.534040692487,
    1.2322657893516,
    .91158310011358,
    1.8788908534927,
    .69811681139453,
    1.1767226952576,
    .98858641944986,
    1.2396486964702,
    1.0822616701665,
    .7753481322096,
    .76860948102754,
    .82419167032501,
    .82704859414698,
    .82559058693092,
    1.1856589638289,
    .75525402187835,
    1.3922568629515,
    .91202394595876,
    .88520111465725,
    .83302439819375,
    1.5372725435314,
    .89177714550085,
    1.2372408530959,
    1.5528101417195,
    1.5366212693726,
    .89043286423929,
    .75705684478985,
    .73518329909511,
    1.0572392569996,
    .93211401184553,
    .75196863280241,
    .69360947719119,
    1.3106012580927,
    1.0209999290642,
    .81133440956099,
    .75559203443239,
    .94831739014224,
    .93322729630976,
    .69417535741881,
    1.2398628905688,
    1.0261547060202,
    .86379439641561,
    .75032871325094,
    1.6696098440084,
    1.8282957404356,
    .70460484132802,
    .79812152719782,
    .8016314317639,
    1.0872600753814,
    1.4987341895633,
    .70529145955969,
    1.0149968053074,
    1.6467690495895,
    1.4677194036031,
    1.3384921199391,
    1.7046600039586,
    1.8412203199988,
    1.4367363864341,
    1.8103264820977,
    2.1242535073081,
    2.1101077640804,
    .745443770473,
    1.3651585515035,
    .94103291060948,
    1.5981517654886,
    .77654854940188,
    .96643611834989,
    .97013991129283,
    .97997241479455,
    1.9501042753844,
    1.1758285978818,
    .9603215961611,
    .96398854110297,
    2.7840981863809,
    1.413576103565,
    1.2364541532613,
    .85434724833597,
    1.3715636637071,
    1.6066244823944,
    1.0686828612843,
    1.3176342918943,
    2.5021204574497,
    1.1330278834213,
    1.1636905462155,
    1.3340487203567,
    1.8517041947194,
    1.1639745295786,
    2.304428659791,
    2.6786336953086,
    2.5354570978302,
    1.1805089049206,
    1.4742162139537,
    2.1143611509083,
    1.5986085606212,
    1.4576906503482,
    1.5624993432094,
    1.3158885079997,
    1.3538646144937,
    1.619414838601,
    1.5015020934095,
    1.3855500128549,
    .86402762365746,
    1.2492579410465,
    1.0127848313981,
    .7158840034754,
    2.6885176805365,
    .83774886448436,
    .81595294371756,
    .70504062914926,
    .73023061883128,
    .69228489060199,
    1.0360869522995,
    .75590017576772,
    .80137295502042,
    .73813943198212,
    .87064014103306,
    .93205385237951,
    .73787688995958,
    .90978243354856,
    1.2648707813937,
    .72349602267222,
    .93927915871284,
    1.0172563832871,
    1.6776109713009,
    1.3175046180712,
    1.1171209329088,
    .78301185980687,
    .78918856044216,
    .74112029520399,
    .74451851348435,
    .76567972163601,
    .87244845911018,
    .90049784898067,
    .76120214907789,
    .71166085065751,
    .90870901548966,
    .70296598825213,
    .84621726620383,
    .69493630035748,
    .75059818136305,
    .87986442867509,
    .86277076414594,
    .87380573440987,
    .80727321528666,
    .70768449827207,
    1.040570485976,
    .78088040381913,
    .74507562843308,
    .69792915007094,
    1.0988781063157,
    1.0027100639643,
    .72261812289435,
    .96179349200945,
    1.0868653716913,
    .86400879397201,
    1.094197020208,
    .74093235691985,
    .67984744050458,
    .8543509596351,
    1.0818096940792,
    .84834713756928,
    1.1852779873098,
    .89126223599075,
    .87108400982232,
    1.1499249237911,
    .75958251908371,
    1.1014705573844,
    .80471862911078,
    .68374044074715,
    1.1058743822208,
    .78261190454279,
    1.1081858646506,
    1.1333032178191,
    .79529240253276,
    1.0984357956297,
    1.0696598625923,
    1.1524457583105,
    1.0091448954756,
    1.2183914737714,
    1.1249902630054,
    .78498039737187,
    1.04667102358,
    .68135274836726,
    1.9934991355412,
    .995276050357,
    .77642057869112,
    .92353879856007,
    1.5308870921603,
    .78757917918954,
    .73146117757714,
    .70335029981046,
    .73869949489316,
    1.6122142696964,
    1.023758116961,
    2.5353634753616,
    2.1057022389201,
    3.0080233992196,
    1.1648858140616,
    1.3964393384387])

fittedvalues_se_colnames = 'fittedvalues_se'.split()

fittedvalues_se_rownames = ['r'+str(n) for n in range(1, 203)]


results = ParamsTableTestBunch(
    params_table=params_table,
    params_table_colnames=params_table_colnames,
    params_table_rownames=params_table_rownames,
    cov=cov,
    cov_colnames=cov_colnames,
    cov_rownames=cov_rownames,
    fittedvalues=fittedvalues,
    fittedvalues_colnames=fittedvalues_colnames,
    fittedvalues_rownames=fittedvalues_rownames,
    fittedvalues_se=fittedvalues_se,
    fittedvalues_se_colnames=fittedvalues_se_colnames,
    fittedvalues_se_rownames=fittedvalues_se_rownames,
    **est
)
