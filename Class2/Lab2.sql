
WITH rowset
AS
(select strftime('%Y', i.invoicedate) as year, ar.name as name,
row_number() over (partition by strftime('%Y', i.invoicedate) order by sum(ii.Quantity) desc)
AS rownum
from invoices i
join invoice_items ii 
	on i.invoiceid = ii.invoiceid 
join tracks t 
	on ii.trackid = t.trackid
join albums al 
	on t.albumid = al.albumid
join artists ar 
	on al.artistid = ar.artistid
where i.billingstate = 'CA'
group by ar.artistid, strftime('%Y', i.invoicedate)
order by strftime('%Y', i.invoicedate), sum(ii.Quantity) desc)

select * from rowset where rownum <= 3
;

