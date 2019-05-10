SELECT  
 reading.ENDTIME endtime, fcb.SERVICEPOINTID spid, reading.DATAVALUE kwh
FROM 
    ITRONEE.servicepointchannel spc
    Join ITRONEE.Readings as reading on
        reading.nodeid = SPC.SPCNODEID
    INNER JOIN ITRONEE.nodelink ON 
        nodelink.rightnodekey = spc.nodekey
    INNER JOIN ITRONEE.servicepoint ON 
        servicepoint.nodekey = nodelink.leftnodekey 
       INNER JOIN itronee.FlatConfigBusiness AS fcb ON
                     fcb.SERVICEPOINTID = ServicePoint.SERVICEPOINTID
  WHERE   reading.endtime BETWEEN '2015-01-01 00:01:00' AND '2018-01-01 00:00:00' 
        AND spc.channelnumber = 1
              and CUSTOMERID = '0102'
			  AND fcb.SERVICEPOINTID NOT LIKE '%SP'
ORDER BY spid, endtime, reading.READINGGROUPALTERNATEKEY
