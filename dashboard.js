/* globals Chart:false, feather:false */

(function () {
  'use strict'

  feather.replace({ 'aria-hidden': 'true' })

  // Graphs
  var ctxHR = document.getElementById('hrChart')
  // eslint-disable-next-line no-unused-vars
  var hrChart = new Chart(ctxHR, {
    type: 'line',
    data: {
      labels: [
        '-115m','-110m','-105m','-100m','-95m','-90m','-85m','80m','-75m','-70m','-65m','-60m','-55m','-50m','-45m','-40m','-35m','-30m','-25m','-20m','-15m','-10m','-5m','now',
      ],
      datasets: [{
        data: [
          72.0, 71.20689369375084, 66.30719126664826, 68.29495111785717, 67.4304145639485, 64.01097924311749, 65.06706161686165, 65.04672497451853, 61.60387757587853, 62.264813346128776, 63.25187361836078, 65.0587410851389, 64.90446852434245, 64.14222722805575, 60.37409950583608, 61.1047350187576, 61.18297609546195, 61.12520062624631, 62.494726232447384, 62.97969797300652, 65.27812026191206, 65.26069769982786, 66.5549112292862, 64.88149788615458, , 64.5549112292862, 63.88149788615458, 63.98149788615458, 80.98149788615458
        ],
        lineTension: 0,
        backgroundColor: 'transparent',
        borderColor: '#177406',
        borderWidth: 3,
        pointBackgroundColor: '#177406'
      }]
    },
    options: {
      title: {
        display: true,
        text: 'Heart Rate'
      },
      scales: {
        yAxes: [{
          ticks: {
            suggestedMin: 30,
            suggestedMax: 130,
            beginAtZero: false
          }
        }]
      },
      legend: {
        display: false
      }
    }
  })



  // Graphs
  var ctxCO2 = document.getElementById('co2Chart')
  // eslint-disable-next-line no-unused-vars
  var co2Chart = new Chart(ctxCO2, {
    type: 'line',
    data: {
      labels: [
        '-115m','-110m','-105m','-100m','-95m','-90m','-85m','80m','-75m','-70m','-65m','-60m','-55m','-50m','-45m','-40m','-35m','-30m','-25m','-20m','-15m','-10m','-5m','now',
      ],
      datasets: [
      { 
        data: [
440.0, 444.74377860866804, 461.5001193764364, 473.71656249388366, 474.72334390553425, 494.0633307174671, 493.7120611319468, 498.25090589642554, 521.4677452945701, 518.5922646528145, 514.2744573773083, 530.6634922270893, 539.9031029378102, 532.5473197822865, 549.071401462904, 558.6794065233217, 555.9957165224866, 569.3527243572904, 585.2119740914554, 582.4529641857429, 581.3956376532404, 597.1111887033161, 593.53379483295, 595.1510238864187        ],
        lineTension: 0,
        backgroundColor: 'transparent',
        borderColor: '#FF5946',
        borderWidth: 3,
        pointBackgroundColor: '#FF5946'
      },
      {
        data: [
444.0, 450.51868973885286, 469.214729914413, 465.92689158208236, 463.74033699569986, 490.8218247699731, 491.43364223675314, 484.4300915764358, 518.4514090113641, 522.5799555487326, 520.0425927424294, 538.4083841686585, 544.1857044784244, 544.9740438032235, 565.3205098496444, 574.8688377357419, 569.1842707042968, 579.6633462635942, 599.026482621453, 600.7282000566482, 613.133391074949, 621.0091806977576, 612.7365010990958, 624.240653860873        ],
        lineTension: 0,
        backgroundColor: 'transparent',
        borderColor: '#FEAAAA',
        borderWidth: 2,
        pointBackgroundColor: '#FEAAAA'
      },
      {
        data: [
448.0, 453.46246813997266, 470.9900244943346, 462.65786078712824, 464.14815603283404, 483.5301125280574, 495.6154217711308, 488.37034902712844, 514.4034681007731, 508.68071769075294, 518.2626388572752, 535.2500012347018, 546.3611763044108, 530.3460463146398, 548.7119115142202, 557.4335135904063, 548.2954886478624, 559.3626918096905, 581.702333156096, 579.8646948938725, 586.1675327855355, 602.0229345732679, 604.6118078234725, 605.1963507212653        ],
        lineTension: 0,
        backgroundColor: 'transparent',
        borderColor: '#FEAAAA',
        borderWidth: 2,
        pointBackgroundColor: '#FEAAAA'
      }
      ]
    },
    options: {
      title: {
        display: true,
        text: 'C02'
      },
      scales: {
        yAxes: [{
          ticks: {
            suggestedMin: 0,
            suggestedMax: 800,
            beginAtZero: false
          }
        }]
      },
      legend: {
        display: false
      }
    }
  })


  // Graphs
  var ctxSPO2 = document.getElementById('spo2Chart')
  // eslint-disable-next-line no-unused-vars
  var spo2Chart = new Chart(ctxSPO2, {
    type: 'line',
    data: {
      labels: [
        '-115m','-110m','-105m','-100m','-95m','-90m','-85m','80m','-75m','-70m','-65m','-60m','-55m','-50m','-45m','-40m','-35m','-30m','-25m','-20m','-15m','-10m','-5m','now',
      ],
      datasets: [
      { 
        data: [
95.73, 95.89763602304077, 94.5856027606819, 95.06520055778066, 95.95743706377073, 93.37847332096078, 91.49358251574664, 89.2020366347337, 91.3184833362058, 91.28477656585727, 89.57526910464478, 88.89259544986277, 88.15116401657835, 85.92967703296979, 87.80566858576533, 89.21125806951865, 88.47372953364003, 87.21959839068823, 88.70858996453276, 85.8926009680988, 85.83612117007615, 84.32939815556858, 85.82486939312408, 86.66598819103518],
        lineTension: 0,
        backgroundColor: 'transparent',
        borderColor: '#FF5946',
        borderWidth: 3,
        pointBackgroundColor: '#FF5946'
      },
      {
        data: [
95.35, 94.52356990505905, 95.08528583827558, 93.93438357747938, 93.7587148554357, 91.92531165426878, 90.87144406732496, 90.22931703234227, 91.08459468689729, 91.74038179170442, 90.6568552335598, 87.95887364850165, 87.79677233077525, 87.81570381680307, 84.31369739456835, 84.75618479052292, 85.88922846168357, 87.92389408607033, 87.6874722270101, 87.04929112396329, 88.64754002247568, 90.31216901935949, 87.92169921058722, 89.08190566141474
],
        lineTension: 0,
        backgroundColor: 'transparent',
        borderColor: '#FEAAAA',
        borderWidth: 2,
        pointBackgroundColor: '#FEAAAA'
      },
      {
        data: [
97.05, 92.59276777546523, 92.76538423870146, 92.81800889379718, 92.2839340957285, 92.52231599552282, 93.08609925281341, 92.47591254719833, 92.15546845910507, 92.04708507433708, 91.59594319996778, 89.76551644621598, 86.0570925180754, 86.13178549254779, 84.31369739456835, 84.75618479052292, 85.88922846168357, 85.84894558615166, 85.53467293341154, 85.8926009680988, 84.73326827916155, 84.32939815556858, 85.82486939312408, 85.77865779876721
      ],
        lineTension: 0,
        backgroundColor: 'transparent',
        borderColor: '#FEAAAA',
        borderWidth: 2,
        pointBackgroundColor: '#FEAAAA'
      }
      ]
    },
    options: {
      title: {
        display: true,
        text: 'SPO2'
      },
      scales: {
        yAxes: [{
          ticks: {
            suggestedMin: 0,
            suggestedMax: 100,
            beginAtZero: false
          }
        }]
      },
      legend: {
        display: false
      }
    }
  })


  // Graphs
  var ctxMIP = document.getElementById('mipChart')
  // eslint-disable-next-line no-unused-vars
  var mipChart = new Chart(ctxMIP, {
    type: 'line',
    data: {
      labels: [
        '-115m','-110m','-105m','-100m','-95m','-90m','-85m','80m','-75m','-70m','-65m','-60m','-55m','-50m','-45m','-40m','-35m','-30m','-25m','-20m','-15m','-10m','-5m','now',
      ],
      datasets: [{
        data: [
 101109.0, 101100.0, 101108.41470984808, 101109.09297426826, 101101.4112000806, 101099.36607527804, 101099.68900039967, 101097.205845018, 101097.94240986135, 101093.73842795513, 101091.20970926457, 101090.75270573051, 101095.40293458637, 101094.63427082, 101092.54131602537, 101094.95320465157, 101092.10085699403, 101081.99821769624, 101078.67006714598, 101075.99378310822, 101066.1739013081, 101067.29453852185, 101071.70927395053, 101068.75745790025       ],
        lineTension: 0,
        backgroundColor: 'transparent',
        borderColor: '#177406',
        borderWidth: 3,
        pointBackgroundColor: '#177406'
      }]
    },
    options: {
      title: {
        display: true,
        text: 'Mask Internal Pressure'
      },
      scales: {
        yAxes: [{
          ticks: {
            suggestedMin: 100500,
            suggestedMax: 101500,
            beginAtZero: false
          }
        }]
      },
      legend: {
        display: false
      }
    }
  })


})()